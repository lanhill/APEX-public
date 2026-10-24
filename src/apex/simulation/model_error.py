"""Model error characterization: well features and the effective simulator.

The effective simulator ``g(m) + r(m, theta)`` runs a file-based simulator
(:class:`~apex.core.simulation.CaseFileSimulator`: ``opm_flow``, ``eclipse``) and adds the
residuals of a model-error model, whose parameters ``theta`` are a state variable the deck
never sees. This module also reads what the PG-LSTM model (``apex.workflows.model_error``)
needs: the wells' perforations and status from the schedule, and the well-block features
of a run from its INIT, restart and summary files.

Ported from PET-fork
-----------------------
PET module : Examples/.../Brugge/ml_model_error.py (``ModelErrorProd``: schedule
             reading, ``get_props_at_wells``, the input sequences of
             ``initialize_lstm_models``) and simulator/eclipse.py (``add_me_residual``),
             the checkout's working copy
PET commit : 1348861b08314b8b3f3e1ccf886c7c095273e202 (not public)
Reference  : Luo, X. Model error characterization for ensemble-based history matching
             through a physics guided long short-term memory network (manuscript).
Differences: the schedule is read with a general record tokenizer (PET-fork splits
             lines at tabs); SOIL is read when the restart has it (PET-fork fails then);
             files are read with resdata instead of PET-fork's ``misc.ecl``.
"""

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from apex.core.errors import SimulationError
from apex.core.simulation import CaseFileSimulator, MemberInput, ModelErrorModel, Simulator
from apex.domain.observations import DataKey
from apex.simulation.deck import report_values

__all__ = [
    "PHASES",
    "RESIDUAL_PREFIX",
    "CaseFiles",
    "ModelErrorSimulator",
    "WellSchedule",
    "feature_names",
    "read_schedule",
    "well_features",
    "well_schedules",
]

#: Static features (INIT), per perforated block.
STATIC = ("DX", "DY", "DZ", "PERMX", "PERMY", "PERMZ", "PORO")
#: The phases a deck may have, with their density and saturation keywords.
PHASES = {"oil": ("OIL_DEN", "SOIL"), "water": ("WAT_DEN", "SWAT"), "gas": ("GAS_DEN", "SGAS")}


def feature_names(phases: Sequence[str]) -> tuple[str, ...]:
    """Return the features per perforated block for a deck's phases, in PET-fork's order.

    The static ones, then each phase's density and saturation (oil, water, gas), with the
    oil saturation, which the simulators do not write, after the others, then PRESS_DIFF
    (the well's WBHP minus the block pressure). Oil-water: ``OIL_DEN WAT_DEN SWAT SOIL
    PRESS_DIFF``; with gas: ``OIL_DEN WAT_DEN SWAT GAS_DEN SGAS SOIL PRESS_DIFF``.

    Raises
    ------
    ValueError
        If ``phases`` lacks oil, repeats a phase or names an unknown one.
    """
    names = [p.lower() for p in phases]
    unknown = sorted(set(names) - set(PHASES))
    if unknown or "oil" not in names or len(set(names)) != len(names):
        raise ValueError(f"phases must include oil, among {sorted(PHASES)}, got {list(phases)}")
    dynamic = ["OIL_DEN"]
    for phase in ("water", "gas"):
        if phase in names:
            dynamic += list(PHASES[phase])
    return (*STATIC, *dynamic, "SOIL", "PRESS_DIFF")


#: Datatype prefix of the residual keys an effective simulator also answers.
RESIDUAL_PREFIX = "MEC:"

#: A quoted item, a record's end ``/`` (also when it ends an item, as in ``100/``), or an
#: item.
_TOKEN = re.compile(r"'[^']*'|/|[^\s/']+")


def _tokens(line: str) -> list[str]:
    """Split a deck line into tokens, dropping a ``--`` comment outside quotes."""
    tokens: list[str] = []
    for token in _TOKEN.findall(line):
        if token.startswith("--"):
            break
        tokens.append(token)
    return tokens


#: Schedule keywords :func:`read_schedule` reads; the others are skipped.
SCHEDULE_KEYWORDS = (
    "COMPDAT", "WCONHIST", "WCONINJH", "WCONPROD", "WCONINJE", "WELOPEN", "TSTEP", "DATES",
)  # fmt: skip


def read_schedule(path: Path) -> list[tuple[str, list[list[str | None]]]]:
    """Return the schedule's keywords of :data:`SCHEDULE_KEYWORDS`, in order, with records.

    A record is its items up to ``/``, quotes stripped, with ``n*`` expanded to ``n``
    defaults (``None``); a block ends at a line that starts with ``/``. ``TSTEP`` is one
    record, whose ``n*v`` items expand to ``n`` copies of ``v``. Text after a record's
    ``/`` and ``--`` comments are ignored.
    """
    lines = path.read_text(errors="replace").splitlines()
    result: list[tuple[str, list[list[str | None]]]] = []
    i = 0
    while i < len(lines):
        tokens = _tokens(lines[i])
        i += 1
        if len(tokens) != 1 or tokens[0] not in SCHEDULE_KEYWORDS:
            continue
        keyword = tokens[0]
        records: list[list[str | None]] = []
        record: list[str | None] = []
        while i < len(lines):
            tokens = _tokens(lines[i])
            i += 1
            if not tokens:
                continue
            if tokens[0] == "/" and not record and keyword != "TSTEP":
                break
            ended = False
            for token in tokens:
                if token == "/":
                    records.append(record)
                    record, ended = [], True
                    break
                star = re.fullmatch(r"(\d*)\*(.*)", token)
                if star and keyword == "TSTEP":
                    record.extend([star.group(2)] * int(star.group(1) or 1))
                elif star and not star.group(2):
                    record.extend([None] * int(star.group(1) or 1))
                else:
                    record.append(token.strip("'"))
            if ended and keyword == "TSTEP":
                break
        result.append((keyword, records))
    return result


@dataclass(frozen=True)
class WellSchedule:
    """One well's perforations and status at each report time.

    Attributes
    ----------
    name, kind
        Well name and ``"producer"`` or ``"injector"``.
    outputs
        Its data types, e.g. ``("WOPR", "WBHP", "WWCT")``.
    blocks
        Every block (1-based ``i, j, k``) it perforates at any report time, first seen first.
    perforated
        ``(n_steps, n_blocks)``: whether each block is perforated at each report time.
    open
        ``(n_steps,)``: whether the well is open at each report time.
    """

    name: str
    kind: str
    outputs: tuple[str, ...]
    blocks: tuple[tuple[int, int, int], ...]
    perforated: NDArray[np.bool_]
    open: NDArray[np.bool_]

    def n_features(self, phases: Sequence[str] = ("oil", "water")) -> int:
        """Return the input features per report time: every feature at every block."""
        return len(feature_names(phases)) * len(self.blocks)


#: Item (0-based) holding the well's status in each well-control keyword.
_STATUS_ITEM = {"WCONHIST": 1, "WCONPROD": 1, "WCONINJH": 2, "WCONINJE": 2, "WELOPEN": 1}


def well_schedules(
    schedule_file: Path | Sequence[Path],
    report_steps: Sequence[int],
    wells: Sequence[tuple[str, str, Sequence[str]]],
) -> list[WellSchedule]:
    """Read each well's perforations and status at the given report steps.

    ``schedule_file`` may be several files, read in order as one schedule (e.g. the deck's
    own SCHEDULE section before its included history). The schedule is walked in order:
    each ``TSTEP`` value and each ``DATES`` record ends a report step (1-based). As in
    Eclipse, ``COMPDAT`` adds connections to a well (a cell named again keeps its place),
    so a well's blocks at a report step are every cell connected so far, open or shut, as
    PET-fork counts them. Its status is that of its last well-control record
    (``WCONHIST``, ``WCONPROD``, ``WCONINJH``, ``WCONINJE``) or well-level ``WELOPEN``. On
    PET-fork's Brugge schedule (every block re-lists every connection) this is its
    reader's ``rpt_idx``-th block.

    Raises
    ------
    ValueError
        If the schedule ends before a report step, or a well has no perforation or status
        there.
    """
    names = {name for name, _, _ in wells}
    cells: dict[str, list[tuple[int, int, int]]] = {}
    status: dict[str, bool] = {}
    snapshots: dict[int, tuple[dict[str, list[tuple[int, int, int]]], dict[str, bool]]] = {}
    wanted, step = set(report_steps), 0

    def advance(n: int) -> None:
        nonlocal step
        for _ in range(n):
            step += 1
            if step in wanted:
                snapshots[step] = ({w: list(c) for w, c in cells.items()}, dict(status))

    files = [schedule_file] if isinstance(schedule_file, Path) else list(schedule_file)
    for keyword, records in (entry for file in files for entry in read_schedule(file)):
        if keyword == "COMPDAT":
            for record in records:
                if record and record[0] in names:
                    if len(record) < 5 or any(v is None for v in record[1:5]):
                        # Defaulted I, J (the WELSPECS location) or K1, K2 are not read.
                        raise ValueError(
                            f"COMPDAT of {record[0]} defaults its I, J, K1 or K2; give them "
                            "explicitly for the model-error model's well blocks"
                        )
                    i, j, k1, k2 = (int(str(v)) for v in record[1:5])
                    connected = cells.setdefault(str(record[0]), [])
                    connected += [
                        (i, j, k) for k in range(k1, k2 + 1) if (i, j, k) not in connected
                    ]
        elif keyword in _STATUS_ITEM:
            item = _STATUS_ITEM[keyword]
            for record in records:
                # Cell or completion items (0 counts as a default, as in Eclipse): a
                # connection's status, not the well's.
                if keyword == "WELOPEN" and any(v not in (None, "0") for v in record[2:]):
                    continue  # its cell stays one of the well's blocks
                if record and record[0] in names:
                    # A defaulted status (1*) is OPEN, Eclipse's default.
                    value = record[item] if len(record) > item else None
                    status[str(record[0])] = value is None or value.lower() == "open"
        elif keyword == "TSTEP":
            advance(len(records[0]) if records else 0)
        else:  # DATES: one report step per date
            advance(len(records))
    missing = sorted(wanted - set(snapshots))
    if missing:
        raise ValueError(
            f"{files[-1].name} has {step} report steps; report steps {missing} are beyond"
        )
    result = []
    for name, kind, outputs in wells:
        per_step: list[list[tuple[int, int, int]]] = []
        opened: list[bool] = []
        for s in report_steps:
            step_cells, step_status = snapshots[s]
            if not step_cells.get(name):
                raise ValueError(f"{name} has no COMPDAT perforation at report step {s}")
            if name not in step_status:
                raise ValueError(f"{name} has no well-control status at report step {s}")
            per_step.append(step_cells[name])
            opened.append(step_status[name])
        union: list[tuple[int, int, int]] = []
        for c in per_step:
            union += [b for b in c if b not in union]
        perforated = np.array([[b in c for b in union] for c in per_step], dtype=bool)
        result.append(
            WellSchedule(name, kind, tuple(outputs), tuple(union), perforated, np.array(opened))
        )
    return result


class CaseFiles:
    """Reads one simulated case's INIT, restart and summary files (ECLIPSE format)."""

    def __init__(self, case: Path) -> None:
        from resdata.grid import Grid  # optional dependency (the reservoir extra)
        from resdata.resfile import ResdataFile
        from resdata.summary import Summary

        self.case = case
        try:
            self.grid = Grid(f"{case}.EGRID")
            self.init = ResdataFile(f"{case}.INIT")
            self.summary = Summary(str(case))
        except (OSError, ValueError) as error:
            raise SimulationError(
                f"cannot read the grid, INIT or summary of {case}: {error}"
            ) from error
        unified = Path(f"{case}.UNRST")
        self._unified = ResdataFile(str(unified)) if unified.exists() else None
        self._file = ResdataFile
        self._vectors: dict[str, NDArray[np.float64]] = {}

    def active(self, block: tuple[int, int, int]) -> int:
        """Return the active index of a 1-based block, ``-1`` if inactive."""
        i, j, k = block
        return int(self.grid.get_active_index(ijk=(i - 1, j - 1, k - 1)))

    def static(self, name: str) -> NDArray[np.float64]:
        """Return an INIT array over the active cells."""
        if name not in self.init:
            raise SimulationError(f"{self.case.name}.INIT has no {name}")
        return np.asarray(self.init[name][0].numpy_view(), dtype=np.float64).copy()

    def restart(self, step: int) -> dict[str, NDArray[np.float64]]:
        """Return the dynamic arrays at report step ``step``."""
        if self._unified is not None:
            view = self._unified.restart_view(report_step=step)
        else:
            path = Path(f"{self.case}.X{step:04d}")
            if not path.exists():
                raise SimulationError(
                    f"{self.case.name} has no restart at report step {step} (the deck needs "
                    "RPTRST BASIC=2 DEN, so that restarts hold the densities)"
                )
            view = self._file(str(path))
        arrays: dict[str, NDArray[np.float64]] = {}
        for name in ("PRESSURE", "SOIL", "SWAT", "SGAS", "OIL_DEN", "WAT_DEN", "GAS_DEN"):
            if name in view:
                arrays[name] = np.asarray(view[name][0].numpy_view(), dtype=np.float64).copy()
        if "PRESSURE" not in arrays:
            raise SimulationError(f"{self.case.name}: no PRESSURE at report step {step}")
        if "SOIL" not in arrays:  # PET-fork: the unwritten saturation is 1 - the others
            others = [arrays[n] for n in ("SWAT", "SGAS") if n in arrays]
            saturation = np.sum(others, axis=0) if others else np.zeros_like(arrays["PRESSURE"])
            arrays["SOIL"] = 1.0 - saturation
        return arrays

    def wbhp(self, well: str, step: int) -> float:
        """Return the well's bottom-hole pressure at report step ``step``."""
        key = f"WBHP:{well}"
        if not self.summary.has_key(key):
            raise SimulationError(f"summary {self.case.name} has no {key}")
        if key not in self._vectors:
            self._vectors[key] = report_values(self.summary, key)
        # report_values follows report_dates, which start at the summary's first report.
        return float(self._vectors[key][step - self.summary.first_report])

    def report_dates(self, steps: Sequence[int]) -> list[str]:
        """Return the ISO dates of report steps (1-based)."""
        dates = list(self.summary.report_dates)
        first = int(self.summary.first_report)
        if min(steps) < first or max(steps) - first >= len(dates):
            raise SimulationError(
                f"{self.case.name} reports steps {first} to {first + len(dates) - 1}, "
                f"not {min(steps)} to {max(steps)}"
            )
        return [dates[s - first].isoformat()[:10] for s in steps]


def well_features(
    files: CaseFiles,
    wells: Sequence[WellSchedule],
    report_steps: Sequence[int],
    phases: Sequence[str] = ("oil", "water"),
) -> list[NDArray[np.float64]]:
    """Return each well's input sequence ``(n_steps, n_features)`` from a simulated case.

    Feature-major, then block (PET-fork's order): every feature of
    :func:`feature_names` at every block of :attr:`WellSchedule.blocks`; a block the well
    does not perforate at a report time (or an inactive one) is 0, PET-fork's padding.

    Raises
    ------
    SimulationError
        If a file lacks a feature (restarts need ``RPTRST BASIC=2 DEN``).
    """
    names = feature_names(phases)
    static = {name: files.static(name) for name in STATIC}
    restarts = [files.restart(step) for step in report_steps]
    for step, arrays in zip(report_steps, restarts, strict=True):
        missing = [n for n in names[len(STATIC) : -1] if n not in arrays]
        if missing:
            raise SimulationError(
                f"{files.case.name}: no {', '.join(missing)} at report step {step} (the deck "
                "needs RPTRST BASIC=2 DEN, so that restarts hold the densities)"
            )
    sequences = []
    for well in wells:
        index = [files.active(b) for b in well.blocks]
        x = np.zeros((len(report_steps), len(names), len(well.blocks)))
        for t, step in enumerate(report_steps):
            bhp = files.wbhp(well.name, step)
            dyn = restarts[t]
            for b, cell in enumerate(index):
                if not well.perforated[t, b] or cell < 0:
                    continue
                x[t, :, b] = [
                    *(static[n][cell] for n in STATIC),
                    *(dyn[n][cell] for n in names[len(STATIC) : -1]),
                    bhp - dyn["PRESSURE"][cell],
                ]
        sequences.append(x.reshape(len(report_steps), -1))
    return sequences


class ModelErrorSimulator:
    """The effective simulator ``g(m) + r(m, theta)``.

    Runs ``inner`` on the member's state without ``theta`` (the deck never sees it), reads
    its summary at the requested keys, adds the model's residuals from the run's files, and
    then lets ``inner`` prune them. Keys ``MEC:<type>`` (:meth:`auxiliary_keys`) give the
    residuals themselves.
    """

    needs_work_dir = True

    def __init__(self, inner: Simulator, model: ModelErrorModel) -> None:
        if not isinstance(inner, CaseFileSimulator):
            raise ValueError(
                f"model error needs a simulator whose runs leave files to read "
                f"(opm_flow, eclipse); {type(inner).__name__} has none"
            )
        self.inner: CaseFileSimulator = inner
        self.model = model

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

    def version(self) -> str | None:
        """Return the inner simulator's version, for provenance."""
        version = getattr(self.inner, "version", None)
        return version() if callable(version) else None

    def auxiliary_keys(self, keys: Sequence[DataKey]) -> list[DataKey]:
        """Return the residual keys the workflow may also request (saved, not assimilated)."""
        return self.model.residual_keys(keys)

    def run(self, member: MemberInput, keys: Sequence[DataKey]) -> NDArray[np.float64]:
        """Return ``g(m) + r(m, theta)`` at ``keys`` (and ``r`` at ``MEC:`` keys)."""
        name = self.model.state_variable
        if name not in member.state:
            raise SimulationError(f"member {member.member_id} has no state variable {name!r}")
        theta = np.asarray(member.state[name], dtype=np.float64)
        # replace(), not a new MemberInput: the member keeps where it runs.
        physical = replace(member, state={k: v for k, v in member.state.items() if k != name})
        data_rows = [i for i, k in enumerate(keys) if not k.datatype.startswith(RESIDUAL_PREFIX)]
        case = self.inner.simulate(physical)
        try:
            # One reading of the run's files gives r at every key: added to the simulated
            # data at the data keys, as is at the MEC: keys.
            values = self.model.residuals(case, theta, keys)
            values[data_rows] += self.inner.read(case, [keys[i] for i in data_rows])
        except BaseException:
            # A failed member's files stay for diagnosis (keep_simulations); only what the
            # inner simulator holds for the case (the geophysics' member model) goes.
            release = getattr(self.inner, "release", None)
            if callable(release):
                release(case)
            raise
        self.inner.clean(case)
        return values
