"""Read an Eclipse-format deck (OPM Flow, Eclipse) into converter facts.

The deck is read from its text, in order, following its ``INCLUDE``s where they stand, so
each keyword is seen in its section:

- ``RUNSPEC``: the grid (``DIMENS``), the units and the phases;
- ``GRID``, ``EDIT``, ``PROPS``: per-cell properties. A keyword whose one record holds a
  value for every cell is an ``array`` (it can carry the state); ``COPY``, ``MULTIPLY``,
  ``ADD`` make ``derived`` properties and ``EQUALS`` ``uniform`` ones. The grid's geometry
  and integer flags (``ACTNUM``) are read but never choosable;
- ``SCHEDULE``: report dates (``DATES``, and ``TSTEP`` counted from ``START``), wells
  (``WELSPECS``, ``COMPDAT``), and the observed history of ``WCONHIST``/``WCONINJH``.

**History.** A real case brings its observations (the user, 2026-10-24): a CSV, a summary
file, or the deck's history keywords, in that order. A history control set at report date
``d_k`` holds (Eclipse keeps a control until it is changed) over ``[d_k, d_k+1)``. The
summary reports that step's mean rate at ``d_k+1``, so that is the observation's time.
A ``SHUT`` well's rates are 0. A history BHP of 0 or defaulted is not an observation.

Nothing is rendered or simulated; a deck with Mako in it is a template, not a deck.
"""

import csv
import datetime as dt
import math
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from apex.core.conversion import CaseFacts, ObservationCandidate, PropertyFact, WellFact
from apex.core.errors import ConfigError
from apex.simulation.deck import (
    _completion,
    _date,
    _expand,
    _include_target,
    _keyword_records,
    _path_aliases,
)

#: The sections of a deck (a keyword alone on its line starts each).
SECTIONS = ("RUNSPEC", "GRID", "EDIT", "PROPS", "REGIONS", "SOLUTION", "SUMMARY", "SCHEDULE")
#: Sections whose per-cell arrays are properties the state may carry.
PROPERTY_SECTIONS = frozenset({"GRID", "EDIT", "PROPS"})
#: Arrays of the grid's geometry and flags: read, never the state.
GEOMETRY = frozenset({"DX", "DY", "DZ", "TOPS", "COORD", "ZCORN", "ACTNUM", "DEPTH",
                      "DXV", "DYV", "DZV", "MAPAXES"})  # fmt: skip
#: Operations that set a property from another (``COPY 'PERMX' 'PERMY' /``) or modify it.
OPERATIONS = frozenset({"COPY", "MULTIPLY", "ADD", "EQUALS", "MINVALUE", "MAXVALUE"})
UNITS = frozenset({"METRIC", "FIELD", "LAB", "PVT-M"})
PHASES = ("OIL", "WATER", "GAS", "DISGAS", "VAPOIL", "CO2STORE", "SOLVENT")
#: History items: WCONHIST's rates (items 4-6) and BHP (item 10); WCONINJH's rate (4), BHP (5).
_PRODUCER_VECTORS = (("WOPR", 3), ("WWPR", 4), ("WGPR", 5))
_INJECTED = {"WATER": "WWIR", "WAT": "WWIR", "GAS": "WGIR", "OIL": "WOIR"}
#: Files over this size are not read (a deck's own arrays are rarely larger).
MAX_FILE_BYTES = 500_000_000
MAX_INCLUDES = 500


@dataclass
class _Keyword:
    keyword: str
    records: list[str]
    section: str
    file: Path


@dataclass
class DeckData:
    """What :func:`read_deck` found: the facts, the arrays and the observed history.

    ``history`` maps ``(date, vector)`` to the observed value; ``arrays`` holds each
    ``array`` property's values in Eclipse cell order, ``i`` fastest.
    """

    facts: CaseFacts
    arrays: dict[str, NDArray[np.float64]] = field(default_factory=dict)
    history: dict[tuple[dt.date, str], float] = field(default_factory=dict)
    files: list[Path] = field(default_factory=list)


#: Whether a caller may read a file (a web-UI user's folders); ``None``: any file.
Allowed = Callable[[Path], bool] | None


def walk(deck: Path, allowed: Allowed = None) -> Iterator[_Keyword]:
    """Yield the deck's keywords in order, ``INCLUDE``s followed where they stand.

    ``allowed`` confines what is read: an include it refuses stops the reading, with the
    same message whether or not the file exists (a web-UI user learns nothing of others'
    files).

    A file may be included more than once (e.g. the same well controls twice in SCHEDULE),
    but not from within itself.

    Raises
    ------
    ConfigError
        If the deck holds Mako (it is a template), includes itself, or has more than
        ``MAX_INCLUDES`` includes.
    """
    root = deck.resolve()
    yield from _walk(root, root, ["RUNSPEC"], [], [0], {}, allowed)


def resolve_include(
    record: str, including: Path, root: Path, aliases: Mapping[str, str]
) -> Path | str | None:
    """Return an INCLUDE record's file, as OPM and Eclipse find it.

    A relative path is taken from the root deck's folder (OPM's ``rootPath``, Eclipse's
    data file folder), else from the including file's folder; ``$ALIAS`` is a ``PATHS``
    alias. A name with an expression left in it, or ``None``, if it cannot be resolved.
    """
    target = _include_target(record, root, root.parent, aliases)
    if isinstance(target, Path) and target.is_file():
        return target
    return _include_target(record, including, root.parent, aliases)


def _walk(
    path: Path,
    root: Path,
    section: list[str],
    chain: list[Path],
    count: list[int],
    aliases: dict[str, str],
    allowed: Allowed = None,
) -> Iterator[_Keyword]:
    if path in chain:
        raise ConfigError(f"{path.name} includes itself")
    count[0] += 1
    if count[0] > MAX_INCLUDES:
        raise ConfigError(f"more than {MAX_INCLUDES} included files")
    chain.append(path)
    text = path.read_text(errors="replace")
    if (
        "<%" in text
        or "${" in text
        or any(line.lstrip().startswith("%") for line in text.splitlines())
    ):
        raise ConfigError(f"{path.name} holds Mako: give the deck, not an APEX template")
    for keyword, records in _keyword_records(text):
        if keyword in SECTIONS:
            section[0] = keyword
            yield _Keyword(keyword, [], keyword, path)
            continue
        if keyword == "PATHS":
            # Aliases of folders ('$FLOC/file'), relative ones from the deck's folder.
            aliases.update(_path_aliases(records, root, root.parent))
            yield _Keyword(keyword, records, section[0], path)
            continue
        if keyword == "INCLUDE" and records:
            target = resolve_include(records[0], path, root, aliases)
            if allowed is not None and isinstance(target, Path) and not allowed(target.resolve()):
                raise ConfigError(f"INCLUDE {records[0][:80]}: not a file of your folders")
            if not isinstance(target, Path) or not target.is_file():
                yield _Keyword("INCLUDE?", records, section[0], path)
                continue
            if target.stat().st_size > MAX_FILE_BYTES:
                yield _Keyword("INCLUDE>", [target.name], section[0], path)
                continue
            yield _Keyword("INCLUDE", [str(target)], section[0], path)
            yield from _walk(target.resolve(), root, section, chain, count, aliases, allowed)
            continue
        yield _Keyword(keyword, records, section[0], path)
    chain.pop()


def _numbers(record: str) -> list[float] | None:
    """Return a record's values as numbers (``n*v`` expanded), or ``None`` if one is not."""
    try:
        return [float(v) for v in _expand(record)]
    except ValueError:
        return None


def read_deck(
    deck: Path, *, skill: str, history: Path | None = None, allowed: Allowed = None
) -> DeckData:
    """Return the facts, arrays and observed history of ``deck`` (see the module).

    ``history`` is a CSV (``time,datatype,value``) or a summary file (``.SMSPEC``/``.UNSMRY``),
    used in place of the deck's own history keywords.

    Raises
    ------
    ConfigError
        If the deck cannot be read: no grid, Mako in it, a self-including file.
    """
    grid: list[int] | None = None
    units: str | None = None
    phases: list[str] = []
    start: dt.date | None = None
    dates: list[dt.date] = []
    properties: dict[str, PropertyFact] = {}
    arrays: dict[str, NDArray[np.float64]] = {}
    heads: dict[str, tuple[int, int]] = {}
    phase_of: dict[str, str] = {}
    cells: dict[str, list[list[int]]] = {}
    kinds: dict[str, str] = {}
    includes: list[str] = []
    files: list[Path] = [deck.resolve()]
    unread: list[str] = []
    # The history: per well, its current controls (they hold until changed), and the wells
    # shut by WELOPEN (their rates are observed as 0 until opened again).
    controls: dict[str, dict[str, float]] = {}
    shut: set[str] = set()
    observed: dict[tuple[dt.date, str], float] = {}
    elapsed = 0.0  # days since START, as TSTEP counts them (fractions kept)

    def step_to(date: dt.date) -> None:
        """Close the report step that ends at ``date``: its controls are observed there."""
        for well, well_controls in controls.items():
            for vector, value in well_controls.items():
                if well in shut:
                    if vector.startswith("WBHP"):
                        continue
                    value = 0.0
                observed[(date, vector)] = value
        dates.append(date)

    for entry in walk(deck, allowed):
        keyword, records = entry.keyword, entry.records
        if keyword == "INCLUDE":
            target = Path(records[0])
            files.append(target)
            includes.append(_relative(target, deck))
        elif keyword == "INCLUDE?":
            unread.append(f"INCLUDE {records[0][:80]}: the file was not found")
        elif keyword == "INCLUDE>":
            limit = MAX_FILE_BYTES // 1_000_000
            unread.append(f"INCLUDE {records[0]}: larger than {limit} MB, not read, and "
                          "copied as it is (keywords in it are not seen)")  # fmt: skip
        elif keyword in ("DIMENS", "SPECGRID") and records and grid is None:
            values = _expand(records[0])[:3]
            if len(values) == 3 and all(v.isdigit() for v in values):
                grid = [int(v) for v in values]
        elif keyword in UNITS and entry.section == "RUNSPEC":
            units = keyword
        elif keyword in PHASES and entry.section == "RUNSPEC":
            phases.append(keyword)
        elif entry.section in PROPERTY_SECTIONS:
            _property(entry, grid, deck, properties, arrays, unread)
        elif keyword == "START" and records:
            start = _date(_expand(records[0]))
        elif entry.section != "SCHEDULE":
            continue
        elif keyword == "DATES":
            for record in records:
                date = _date(_expand(record))
                if date is None:
                    unread.append(f"DATES {record[:40]}: not a date")
                    continue
                if start is not None:
                    elapsed = float((date - start).days)
                step_to(date)
        elif keyword == "TSTEP":
            if start is None:
                unread.append("TSTEP without START: its times are unknown")
                continue
            for record in records:
                for days in _numbers(record) or []:
                    elapsed += days
                    if not elapsed.is_integer() and "fractional" not in " ".join(unread):
                        unread.append("TSTEP reaches fractional days: observations there are "
                                      "dated by their day")  # fmt: skip
                    step_to(start + dt.timedelta(days=int(elapsed)))
        elif keyword == "WELSPECS":
            for record in records:
                fields = _expand(record)
                if len(fields) >= 4 and fields[2].isdigit() and fields[3].isdigit():
                    heads[fields[0]] = (int(fields[2]), int(fields[3]))
                if len(fields) >= 6:
                    phase_of[fields[0]] = fields[5].upper()
        elif keyword == "COMPDAT":
            for record in records:
                found = _completion(_expand(record), heads)
                if found is None:
                    unread.append(f"COMPDAT {record[:60]}: not read")
                    continue
                name, completed = found
                known = cells.setdefault(name, [])
                known.extend(c for c in completed if c not in known)
        elif keyword in ("WCONHIST", "WCONPROD", "WCONINJH", "WCONINJE", "WELOPEN"):
            for record in records:
                fields = _expand(record)
                if not fields:
                    continue
                well = fields[0]
                if "*" in well or "?" in well:
                    unread.append(f"{keyword} {record[:50]}: a well name pattern, not followed")
                    continue
                if keyword == "WELOPEN":
                    status = fields[1].upper() if len(fields) > 1 else "OPEN"
                    if any(f != "*" for f in fields[2:7]):
                        # Connections opened or shut: the well's controls are unchanged.
                        unread.append(f"WELOPEN {record[:50]}: connections, not the well")
                        continue
                    if status in ("SHUT", "STOP"):
                        shut.add(well)
                    elif status == "OPEN":
                        shut.discard(well)
                    continue
                kinds[well] = "producer" if keyword in ("WCONHIST", "WCONPROD") else "injector"
                shut.discard(well)
                if keyword == "WCONHIST":
                    controls[well] = _producer_history(fields)
                elif keyword == "WCONINJH":
                    controls[well] = _injector_history(fields)
                else:
                    controls.pop(well, None)  # no longer under history control
    if grid is None:
        raise ConfigError(f"{deck.name}: no DIMENS or SPECGRID, so the grid is unknown")
    source = "deck_history"
    if history is not None:
        observed, source = _read_history(history), _history_source(history)
    wells = {
        name: WellFact(name=name, kind=kinds.get(name, "unknown"),  # type: ignore[arg-type]
                       cells=cells.get(name, []), phase=phase_of.get(name))
        for name in sorted(set(heads) | set(cells) | set(kinds))
    }  # fmt: skip
    facts = CaseFacts(
        format="eclipse_deck", skill=skill, source=deck.name, grid=grid,
        cells=int(np.prod(grid)), units=units, phases=phases, start=start,
        report_dates=sorted(set(dates)), properties=properties, wells=wells,
        candidates=_candidates(observed, source), includes=includes, unread=unread[:40],
    )  # fmt: skip
    return DeckData(facts, arrays, observed, files)


def _relative(path: Path, deck: Path) -> str:
    try:
        return str(path.resolve().relative_to(deck.resolve().parent))
    except ValueError:
        return path.name


def _property(
    entry: _Keyword,
    grid: list[int] | None,
    deck: Path,
    properties: dict[str, PropertyFact],
    arrays: dict[str, NDArray[np.float64]],
    unread: list[str],
) -> None:
    """Record a GRID/EDIT/PROPS keyword: an array, or an operation on properties."""
    keyword, records = entry.keyword, entry.records
    file = _relative(entry.file, deck)
    if keyword in OPERATIONS:
        for record in records:
            fields = _expand(record)
            if not fields:
                continue
            name = fields[0].upper()
            target = fields[1].upper() if keyword == "COPY" and len(fields) >= 2 else name
            box = any(f != "*" for f in fields[2:8])
            where = " in a box" if box else ""
            if keyword == "COPY":
                properties[target] = PropertyFact(
                    name=target, source="derived", choosable=False, derived_from=name,
                    file=file, reason=f"copied from {name}{where}: it follows {name}",
                )  # fmt: skip
            elif keyword == "EQUALS" and target not in properties:
                value = _numbers(fields[1]) if len(fields) > 1 and not box else None
                properties[target] = PropertyFact(
                    name=target, source="uniform", choosable=False, file=file,
                    minimum=value[0] if value else None, maximum=value[0] if value else None,
                    mean=value[0] if value else None,
                    reason=f"set by EQUALS{where}: give it as an array to make it the state",
                )  # fmt: skip
            elif target in properties and properties[target].source == "array":
                # The state would be changed after it is written, or overwritten: refused.
                _unchoosable(properties, target, f"{keyword}{where} changes it after it is set")
        return
    if len(records) != 1 or grid is None:
        return
    values = _numbers(records[0])
    if values is None:
        return
    size = int(np.prod(grid))
    if len(values) != size and keyword in properties and properties[keyword].source == "array":
        # Set again over a BOX: the state would be overwritten there.
        _unchoosable(properties, keyword, "it is set again over a box after it is set")
        return
    if len(values) != size:
        # PROPS holds tables (PVTW, ...): only GRID's and EDIT's short arrays are boxes.
        if keyword not in GEOMETRY and len(values) > 1 and entry.section in ("GRID", "EDIT"):
            unread.append(f"{keyword}: {len(values)} values for {size} cells (a box?), not read")
        return
    array = np.asarray(values, dtype=np.float64)
    if keyword in properties and properties[keyword].source == "array":
        # A second definition would overwrite the member's state.
        _unchoosable(properties, keyword, "it is set more than once in the deck")
        return
    reason = None
    if keyword in GEOMETRY:
        reason = "the grid's geometry or flags, not a property to match"
    arrays[keyword] = array
    # Cells set to 0 (inactive, or barriers) stay 0: the statistics are of the others.
    set_ = array[array != 0]
    if set_.size == 0 and reason is None:
        reason = "every cell is 0"
    properties[keyword] = PropertyFact(
        name=keyword, source="array", choosable=reason is None, reason=reason, file=file,
        zero_cells=int(array.size - set_.size),
        minimum=float(set_.min()) if set_.size else 0.0,
        maximum=float(set_.max()) if set_.size else 0.0,
        mean=float(set_.mean()) if set_.size else 0.0,
    )  # fmt: skip


def _unchoosable(properties: dict[str, PropertyFact], name: str, reason: str) -> None:
    properties[name] = properties[name].model_copy(update={"choosable": False, "reason": reason})


def _item(fields: list[str], index: int) -> float | None:
    """Return a record's item ``index`` (0-based) as a number, or ``None`` if defaulted."""
    if len(fields) <= index or fields[index] in ("*", ""):
        return None
    try:
        return float(fields[index])
    except ValueError:
        return None


def _producer_history(fields: list[str]) -> dict[str, float]:
    """Return a WCONHIST record's observed vectors (a SHUT or STOP well's rates are 0)."""
    well = fields[0]
    open_ = len(fields) < 2 or fields[1].upper() not in ("SHUT", "STOP")
    found = {
        f"{vector}:{well}": (_item(fields, index) or 0.0) if open_ else 0.0
        for vector, index in _PRODUCER_VECTORS
    }
    bhp = _item(fields, 9)
    if open_ and bhp is not None and bhp > 0:
        found[f"WBHP:{well}"] = bhp
    return found


def _injector_history(fields: list[str]) -> dict[str, float]:
    """Return a WCONINJH record's observed vectors."""
    well = fields[0]
    vector = _INJECTED.get(fields[1].upper() if len(fields) > 1 else "", "WWIR")
    open_ = len(fields) < 3 or fields[2].upper() not in ("SHUT", "STOP")
    found = {f"{vector}:{well}": (_item(fields, 3) or 0.0) if open_ else 0.0}
    bhp = _item(fields, 4)
    if open_ and bhp is not None and bhp > 0:
        found[f"WBHP:{well}"] = bhp
    return found


def _history_source(path: Path) -> str:
    return "csv" if path.suffix.lower() == ".csv" else "summary"


def _read_history(path: Path) -> dict[tuple[dt.date, str], float]:
    """Return the observations of a CSV (``time,datatype,value``) or a summary file."""
    if path.suffix.lower() == ".csv":
        return _read_csv(path)
    if path.suffix.upper() in (".SMSPEC", ".UNSMRY", ".FSMSPEC", ".FUNSMRY", ""):
        return _read_summary(path)
    raise ConfigError(f"{path.name}: give the history as a CSV or a summary file (.SMSPEC)")


def _read_csv(path: Path) -> dict[tuple[dt.date, str], float]:
    observed: dict[tuple[dt.date, str], float] = {}
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        columns = {c.strip().lower(): c for c in reader.fieldnames or []}
        time = columns.get("time") or columns.get("date")
        vector = columns.get("datatype") or columns.get("vector")
        value = columns.get("value")
        if not (time and vector and value):
            raise ConfigError(f"{path.name}: the history CSV needs columns time, datatype, value")
        for line, row in enumerate(reader, start=2):
            try:
                date = dt.date.fromisoformat(row[time].strip()[:10])
                number = float(row[value])
            except ValueError as error:
                raise ConfigError(f"{path.name}, line {line}: {error}") from None
            if math.isfinite(number):
                observed[(date, row[vector].strip())] = number
    return observed


def _read_summary(path: Path) -> dict[tuple[dt.date, str], float]:
    """Return a summary file's well vectors at its report dates (the reservoir extra)."""
    try:
        from resdata.summary import Summary
    except ImportError as error:  # pragma: no cover - the reservoir extra is missing
        raise ConfigError("reading a summary file needs the reservoir extra (resdata)") from error
    from apex.simulation.deck import report_values

    case = path.with_suffix("") if path.suffix else path
    try:
        summary = Summary(str(case))
    except (OSError, ValueError) as error:
        raise ConfigError(f"{path.name}: not a summary file ({error})") from None
    report = [d.date() if isinstance(d, dt.datetime) else d for d in summary.report_dates]
    observed: dict[tuple[dt.date, str], float] = {}
    for prefix in ("WOPR", "WWPR", "WGPR", "WWIR", "WGIR", "WBHP"):
        for key in summary.keys(f"{prefix}:*"):
            values = report_values(summary, key)
            for date, value in zip(report, values, strict=False):
                observed[(date, key)] = float(value)
    return observed


def _candidates(
    observed: dict[tuple[dt.date, str], float], source: str
) -> list[ObservationCandidate]:
    by_vector: dict[str, list[tuple[dt.date, float]]] = {}
    for (date, vector), value in observed.items():
        by_vector.setdefault(vector, []).append((date, value))
    found = []
    for vector, points in sorted(by_vector.items()):
        points.sort()
        found.append(ObservationCandidate(
            vector=vector, source=source, times=len(points),  # type: ignore[arg-type]
            first=points[0][0], last=points[-1][0],
            largest=max(abs(v) for _, v in points),
        ))  # fmt: skip
    return found
