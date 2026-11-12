"""Shared parts of deck-based reservoir simulator adapters (OPM Flow, Eclipse).

Every member's deck is rendered from a Mako template into its own work directory. The
template receives every state variable by name (a 1-D array), plus ``member_id`` and
``case_dir`` (for ``INCLUDE`` paths: the template's directory, or the simulator's
``case_dir`` option, e.g. a copied example's own folder while its template stays in the
examples root). Results are read from the
Eclipse-format summary files (``.SMSPEC`` with ``.UNSMRY`` or ``.Snnnn``) at report dates.
Each :class:`DataKey` is ``(ISO date, "<VECTOR>:<WELL>")``, e.g.
``("2024-03-11", "WOPR:PRO1")``. Predictions read whole series instead (:func:`read_series`).
This needs the ``reservoir`` extra (``mako``, ``resdata``).
"""

import datetime as dt
import fnmatch
import itertools
import os
import re
import shlex
import shutil
import warnings
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, cast

import numpy as np
from numpy.typing import NDArray

from apex.core.errors import SimulationError, TransientSimulationError
from apex.core.simulation import MemberInput, MemberSeries, SeriesRequest, match_vectors
from apex.domain.observations import DataKey

if TYPE_CHECKING:
    from resdata.summary import Summary

#: Base name of every member's deck (``MODEL.DATA``) and of its outputs.
DECK_NAME = "MODEL"

#: The simulators built on this module: they run the same deck template, so a study may
#: run a case's deck on either (``[study] simulator``).
DECK_SIMULATORS = frozenset({"opm_flow", "eclipse"})
#: Their options that mean the same to each; the others (``executable``, ``flow_args``,
#: ``eclrun_args``, ``version``, ``licenses``) belong to one simulator and are dropped when a
#: study switches.
PORTABLE_OPTIONS = frozenset(
    {"deck_template_file", "template_values", "keep_files", "timeout_s", "levels",
     "transient_patterns", "case_dir"}
)  # fmt: skip


def render_deck(
    template_path: Path,
    member: MemberInput,
    template_values: Mapping[str, object],
    case_dir: Path | None = None,
) -> str:
    """Render the deck template for ``member``; ``${case_dir}`` is ``case_dir`` if given.

    Raises
    ------
    SimulationError
        If the template fails to render (e.g. an undefined name).
    """
    # Imported here: mako is an optional dependency, and the compiled template is not kept
    # on the adapter so that the adapter stays picklable for process pools.
    from mako.template import Template

    values = {
        **template_values,
        **{name: np.asarray(v) for name, v in member.state.items()},
        "member_id": member.member_id,
        "case_dir": str(case_dir or template_path.resolve().parent),
    }
    try:
        return str(Template(filename=str(template_path)).render(**values))
    except Exception as error:
        raise SimulationError(
            f"deck template failed for member {member.member_id}: {error}"
        ) from error


#: Mako constructs that give a template names its own parse cannot see: another template's,
#: or values reached through the context.
_DYNAMIC_NAMES = re.compile(
    r"<%\s*(include|namespace|inherit)\b|\bcontext\s*[\[.]|\blocals\(|\bglobals\("
)


def template_names(template_path: Path) -> frozenset[str] | None:
    """Return the names a deck template reads from its values, or ``None`` if unknown.

    Every identifier its expressions, control lines and code blocks use without defining
    it (a superset: a loop's own variable may be among them). ``None`` when the template
    includes or inherits another, reaches its values through ``context``, or cannot be
    parsed: then which values it reads cannot be told from its text.
    """
    from mako.exceptions import MakoException
    from mako.lexer import Lexer

    try:
        text = template_path.read_text(errors="replace")
        if _DYNAMIC_NAMES.search(text):
            return None
        root = Lexer(text).parse()
    except (OSError, MakoException, SyntaxError):  # unreadable or unparsable: unknown
        return None
    names: set[str] = set()
    pending = [root]
    while pending:
        node = pending.pop()
        undeclared = getattr(node, "undeclared_identifiers", None)
        if callable(undeclared):
            names |= set(undeclared())
        pending.extend(getattr(node, "nodes", ()))
    return frozenset(names)


def templates_names(paths: Sequence[Path]) -> frozenset[str] | None:
    """Return the names any of several deck templates reads, or ``None`` if one is unknown.

    A multilevel deck's levels may render templates of their own (:class:`FidelityLevels`):
    a value one of them reads is read.
    """
    found: set[str] = set()
    for path in paths:
        names = template_names(path)
        if names is None:
            return None
        found |= names
    return frozenset(found)


#: A path a deck template names under its case folder: ``${case_dir}/include/Grid``. It
#: ends at a quote, a space, a comma or a brace: GEOS lists files as
#: ``"{ ${case_dir}/pvdo.txt, ${case_dir}/pvtw.txt }"``, where they are no part of a name.
#: Parentheses and semicolons may be (``PERM(1).GRDECL``). A computed path
#: (``grid_${level}.inc``, ``f(${i}).inc``) keeps its ``$`` and so is recognised.
_CASE_PATH = re.compile(r"\$\{case_dir\}/([^'\"\s,{}]+)")


def case_dir_files(template_path: Path) -> list[str] | None:
    """Return the paths a deck template names under ``${case_dir}``, relative to it.

    What a remote host needs of the case folder, instead of the whole folder
    (which may hold results and runs). ``None`` when a path is itself computed by the
    template (``${case_dir}/grid_${level}.inc``): then the whole folder is needed.
    """
    found = [m.rstrip("/") for m in _CASE_PATH.findall(template_path.read_text(errors="replace"))]
    if any("$" in path or ".." in Path(path).parts for path in found):
        return None
    return sorted(set(found))


def report_values(summary: "Summary", key: str) -> NDArray[np.float64]:
    """Return a summary vector at the end of each report step, in ``report_dates`` order.

    The values of the report steps' last ministeps, not interpolated: resdata's
    ``numpy_vector(report_only=True)`` interpolates at the report *dates* (midnight), which
    differ from the report times when a step ends within a day (e.g. Brugge's TSTEPs).
    Summary files hold float32; the values are widened, never rounded again.

    Raises
    ------
    SimulationError
        If the report steps and report dates do not match one to one.
    """
    full = np.asarray(summary.numpy_vector(key), dtype=np.float64)
    with warnings.catch_warnings():  # resdata deprecates its own vector type inside the call
        warnings.simplefilter("ignore", DeprecationWarning)
        index = np.array(list(summary.report_index_list()), dtype=np.intp)
    if index.size != len(summary.report_dates):
        raise SimulationError(
            f"summary has {index.size} report steps but {len(summary.report_dates)} report dates"
        )
    return full[index]


def read_summary(case: Path, keys: Sequence[DataKey]) -> NDArray[np.float64]:
    """Read summary vectors of ``case`` (path without extension) at report dates.

    Raises
    ------
    SimulationError
        If a key is a field component, the summary cannot be loaded, a vector is missing,
        or a date is not a report step.
    """
    fields = sorted({f"{k.datatype}@{k.time}" for k in keys if k.index is not None})
    if fields:
        raise SimulationError(
            f"deck simulators read summary vectors only, not fields: {', '.join(fields)}"
        )
    from resdata.summary import Summary  # optional dependency

    try:
        summary = Summary(str(case))
    except (OSError, ValueError) as error:
        raise SimulationError(f"cannot read summary {case}: {error}") from error
    report_dates = {
        d.date() if isinstance(d, dt.datetime) else d: i for i, d in enumerate(summary.report_dates)
    }
    vectors: dict[str, NDArray[np.float64]] = {}
    values = np.empty(len(keys))
    for i, key in enumerate(keys):
        try:
            date = dt.date.fromisoformat(key.time)
        except ValueError as error:
            raise SimulationError(f"report time {key.time!r} is not an ISO date") from error
        if date not in report_dates:
            raise SimulationError(f"{key.time} is not a report date of {case.name}")
        if key.datatype not in vectors:
            if not summary.has_key(key.datatype):
                raise SimulationError(f"summary {case.name} has no vector {key.datatype}")
            vectors[key.datatype] = report_values(summary, key.datatype)
        values[i] = vectors[key.datatype][report_dates[date]]
    return values


def read_series(case: Path, request: SeriesRequest) -> MemberSeries:
    """Read the requested summary vectors of ``case`` at its report dates.

    Raises
    ------
    SimulationError
        If the summary cannot be loaded, a pattern matches no vector, or a requested time
        is not a report date.
    """
    from resdata.summary import Summary  # optional dependency

    try:
        summary = Summary(str(case))
    except (OSError, ValueError) as error:
        raise SimulationError(f"cannot read summary {case}: {error}") from error
    try:
        vectors = match_vectors(request.vectors, list(summary.keys()))
    except ValueError as error:
        raise SimulationError(f"summary {case.name}: {error}") from error
    dates = [
        (d.date() if isinstance(d, dt.datetime) else d).isoformat() for d in summary.report_dates
    ]
    repeated = sorted({d for d in dates if dates.count(d) > 1})
    if repeated:
        # Report steps within one day would share a label, and members could then be
        # read at different steps; refuse rather than guess.
        raise SimulationError(
            f"summary {case.name} has several report steps on {', '.join(repeated)}; "
            "predictions need one report step per day"
        )
    if request.times is None:
        columns = list(range(len(dates)))
    else:
        index = {date: i for i, date in enumerate(dates)}
        missing = [t for t in request.times if t not in index]
        if missing:
            raise SimulationError(f"{', '.join(missing)}: not report dates of {case.name}")
        columns = [index[t] for t in request.times]
    values = np.array([report_values(summary, v) for v in vectors])[:, columns]
    return MemberSeries(tuple(dates[c] for c in columns), vectors, values)


def is_summary_file(path: Path) -> bool:
    """Whether ``path`` is a summary file: ``.SMSPEC``, ``.UNSMRY``, ``.ESMRY`` or ``.Snnnn``."""
    suffix = path.suffix.upper()
    return suffix in {".SMSPEC", ".UNSMRY", ".ESMRY"} or (
        len(suffix) == 6 and suffix.startswith(".S") and suffix[2:].isdigit()
    )


def prune(work_dir: Path, keep: Callable[[Path], bool]) -> None:
    """Delete everything in ``work_dir`` that ``keep`` does not keep (to save disk space)."""
    for path in work_dir.iterdir():
        if not keep(path):
            shutil.rmtree(path) if path.is_dir() else path.unlink()


def tail(path: Path, lines: int = 20) -> str:
    """Return the last ``lines`` lines of a log file (or a placeholder if unreadable)."""
    try:
        return "\n".join(path.read_text(errors="replace").splitlines()[-lines:])
    except OSError:
        return "(no log)"


#: Output of a failed run that means the failure may not recur, so the executor retries it
#: (case-insensitive regular expressions). Licence servers and checkouts (FlexNet/FLEXlm,
#: as Eclipse uses), and file systems that briefly fail (NFS stale handles, I/O errors),
#: fail runs that succeed a minute later, as does Open MPI's ``MPI_Init`` when it fails
#: before the simulator reads its deck (after PMIx failed to open its shared-memory file:
#: two Brugge members on a cluster host, 2026-10-14). The PMIx message itself is not listed:
#: runs that succeed can print it too, so a later deck error would look transient. A
#: simulator's own failures (convergence, deck errors) are not listed: they repeat
#: identically.
TRANSIENT_PATTERNS: tuple[str, ...] = (
    r"licen[cs]e.{0,40}(not available|unavailable|checkout|check out|denied|server|expired)",
    r"flex(net|lm)",
    r"lmgrd",
    r"cannot connect to (the )?licen[cs]e",
    r"stale (nfs )?file handle",
    r"input/output error",
    r"resource temporarily unavailable",
    r"an error occurred in mpi_init",  # Open MPI's wording; also MPI_Init_thread
)


def failure_error(
    message: str, returncode: int | None, log: Path, extra_patterns: Sequence[str] = ()
) -> SimulationError:
    """Return the error of a failed run: transient if it looks like one, else permanent.

    Transient when the process was killed by a signal (a negative return code: e.g. the
    kernel's out-of-memory killer), or when the log matches :data:`TRANSIENT_PATTERNS` or
    one of ``extra_patterns`` (a simulator's ``transient_patterns`` option).
    """
    killed = returncode is not None and returncode < 0
    try:
        text = log.read_text(errors="replace")[-20_000:]
    except OSError:
        text = ""
    patterns = (*TRANSIENT_PATTERNS, *extra_patterns)
    matched = next((p for p in patterns if re.search(p, text, re.IGNORECASE)), None)
    if killed or matched is not None:
        reason = (
            f"killed by signal {-(returncode or 0)}" if killed else f"output matches {matched!r}"
        )
        return TransientSimulationError(f"{message} [transient: {reason}]")
    return SimulationError(message)


class FidelityLevels:
    """Which fidelity level a member runs at, and what its deck gets.

    ``levels`` holds one table per level, in member-id order: ``first`` (its first member
    id) and ``size``, optionally ``deck_template_file`` (a coarse deck of its own) and
    ``upscale`` (grid coarsening, :func:`apex.simulation.upscaling.coarsen`: ``method``,
    ``dim`` (``(ny, nx)``), ``max_trunc`` and optionally ``state`` (the field ``haar``
    follows; all its layers), ``layers`` (``nz``, else from the field's size), ``max_diff``,
    ``wells``, ``radius``). The deck template receives ``level`` (``-1``
    without levels, as SimulatorWrap) and ``coarse``, the member's ``COARSEN`` records
    (empty without upscaling).

    Raises
    ------
    ValueError
        If the levels overlap, or an upscaling is incomplete.
    """

    def __init__(self, levels: Sequence[Mapping[str, object]] | None, default: Path) -> None:
        self.default = default
        self.levels = [dict(level) for level in (levels or [])]
        expected = 0
        for index, level in enumerate(self.levels):
            first, size = int(str(level.get("first", -1))), int(str(level.get("size", 0)))
            if first != expected or size < 1:
                raise ValueError(f"level {index} must start at member {expected} with size >= 1")
            expected = first + size
            template = level.get("deck_template_file")
            if template is not None:
                path = Path(str(template)).resolve()
                if not path.is_file():
                    raise ValueError(f"level {index}: deck template not found: {path}")
                level["deck_template_file"] = path
            upscale = level.get("upscale")
            if upscale is not None:
                options = dict(upscale)  # type: ignore[call-overload]
                if "dim" not in options or "max_trunc" not in options:
                    raise ValueError(f"level {index}: upscale needs dim and max_trunc")
                if options.get("method", "uniform") == "haar" and "state" not in options:
                    raise ValueError(f"level {index}: haar upscaling needs the state it follows")
                if len(options.get("wells", ())) != len(options.get("radius", ())):
                    raise ValueError(f"level {index}: upscale needs one radius per well")
                level["upscale"] = options

    @property
    def upscales(self) -> bool:
        """Whether any level coarsens the grid (Eclipse ``COARSEN``)."""
        return any(level.get("upscale") is not None for level in self.levels)

    def level_of(self, member_id: int) -> int:
        """Return the member's level, ``-1`` without levels (or for a mean-model run)."""
        for index, level in enumerate(self.levels):
            first, size = int(str(level["first"])), int(str(level["size"]))
            if first <= member_id < first + size:
                return index
        return -1

    def templates(self) -> list[Path]:
        """Return every deck template the levels render: the default, then levels' own."""
        own = [Path(str(t)) for t in (lv.get("deck_template_file") for lv in self.levels) if t]
        return [self.default, *own]

    def for_member(self, member: MemberInput) -> tuple[Path, dict[str, object]]:
        """Return the member's deck template and its extra template values."""
        from apex.simulation.upscaling import coarsen

        index = self.level_of(member.member_id)
        if index < 0:
            return self.default, {"level": -1, "coarse": []}
        level = self.levels[index]
        template = level.get("deck_template_file") or self.default
        coarse: list[tuple[int, ...]] = []
        upscale = level.get("upscale")
        if isinstance(upscale, dict):
            dim = [int(n) for n in upscale["dim"]]
            layers = int(upscale.get("layers", 1))
            state = upscale.get("state")
            if state is not None:
                # The whole field (every layer); coarsen checks its size.
                field = np.asarray(member.state[str(state)], dtype=np.float64)
                if "layers" not in upscale:
                    layers = max(1, field.size // (dim[0] * dim[1]))
            else:
                field = np.zeros(dim[0] * dim[1] * layers)
            coarse = list(
                coarsen(
                    field,
                    dim,
                    method=upscale.get("method", "uniform"),
                    max_trunc=float(upscale["max_trunc"]),
                    max_diff=float(upscale.get("max_diff", 0.0)),
                    wells=upscale.get("wells", ()),
                    radius=upscale.get("radius", ()),
                    layers=layers,
                )
            )
        assert isinstance(template, Path)
        return template, {"level": index, "coarse": coarse}


# ---------------------------------------------------------------------- reading a deck

#: Included files larger than this are not scanned for wells (grid geometry files).
_SCAN_MAX_BYTES = 5_000_000
_KEYWORD = re.compile(r"^([A-Z][A-Z0-9_]{0,7})\s*$")
#: A section keyword with other text after it on its line ("PROPS  ====="): Eclipse reads a
#: keyword's first 8 columns only, so the rest is a comment (a ruler).
_SECTION_RULER = re.compile(r"^(RUNSPEC|GRID|EDIT|PROPS|REGIONS|SOLUTION|SUMMARY|SCHEDULE)\s+\S")
_INCLUDE_PATH = re.compile(r"""['"]([^'"]+)['"]|(\S+)""")


def deck_layout(template_path: Path, case_dir: Path | None = None) -> dict[str, object]:
    """Return a deck template's grid and well completions, read from its text.

    Reads ``DIMENS`` (or ``SPECGRID``), ``WELSPECS`` and ``COMPDAT`` from the template and
    the files it includes (``${case_dir}`` is ``case_dir``, by default the template's
    folder; included files over
    5 MB are skipped). Nothing is rendered or simulated; Mako lines are ignored.

    Returns
    -------
    dict
        ``grid``: ``[nx, ny, nz]`` or ``None``; ``wells``: well name to its completions'
        1-based Eclipse cells ``[[i, j, k], ...]``, in deck order (the format of distance
        localization's ``positions``); ``files``: the files read; ``skipped``: COMPDAT
        records that could not be read (e.g. Mako expressions).
    """
    grid: list[int] | None = None
    heads: dict[str, tuple[int, int]] = {}
    wells: dict[str, list[list[int]]] = {}
    skipped: list[str] = []
    files: list[str] = []
    for path in _deck_files(template_path, case_dir):
        files.append(str(path))
        for keyword, records in _keyword_records(path.read_text(errors="replace")):
            if keyword in ("DIMENS", "SPECGRID") and records and grid is None:
                values = _expand(records[0])[:3]
                if len(values) == 3 and all(v.isdigit() for v in values):
                    grid = [int(v) for v in values]
            elif keyword == "WELSPECS":
                for record in records:
                    fields = _expand(record)
                    if len(fields) >= 4 and fields[2].isdigit() and fields[3].isdigit():
                        heads[fields[0]] = (int(fields[2]), int(fields[3]))
            elif keyword == "COMPDAT":
                for record in records:
                    cells = _completion(_expand(record), heads)
                    if cells is None:
                        skipped.append(record)
                        continue
                    name, found = cells
                    known = wells.setdefault(name, [])
                    known.extend(c for c in found if c not in known)
    return {"grid": grid, "wells": wells, "files": files, "skipped": skipped[:20]}


def deck_keywords(template_path: Path, case_dir: Path | None = None) -> list[tuple[str, list[str]]]:
    """Return a deck's keywords and their ``/``-terminated records, its includes followed.

    In the order the files are read (the template, then each included file, each once);
    comments dropped; a table's rows up to its ``/`` are one record. For readers of a
    deck's tables (e.g. ``SWOF``, ``PVTW``) outside the simulators.
    """
    found: list[tuple[str, list[str]]] = []
    for path in _deck_files(template_path, case_dir):
        found.extend(_keyword_records(path.read_text(errors="replace")))
    return found


def schedule_keywords(
    template_path: Path, case_dir: Path | None = None
) -> tuple[list[tuple[str, list[str]]], list[str]]:
    """Return the keywords of a deck's ``SCHEDULE`` section in deck order, and notes.

    Includes are expanded where they stand (unlike :func:`deck_keywords`), so ``DATES``,
    ``TSTEP`` and the wells' controls come in the order the simulator reads them; reading
    stops at ``END``. ``START`` is included from wherever it stands. The notes say what could
    not be followed (Mako lines, includes of template expressions), for readers that build a
    schedule from the text (wells' controls by period).
    """
    notes: list[str] = []

    def note(message: str) -> None:
        if message not in notes:
            notes.append(message)

    found: list[tuple[str, list[str]]] = []
    scheduled = False
    templated = False
    for keyword, records, mako in _ordered_records(template_path, case_dir, note):
        if keyword == "SCHEDULE":
            scheduled = True
        elif keyword in _SECTIONS:
            scheduled = False
        if scheduled or keyword == "START":
            found.append((keyword, records))
            if mako and records and not templated:
                templated = True
                note(f"{keyword} is in a file with Mako lines: its records may differ")
    return found, notes


def _deck_files(template_path: Path, case_dir: Path | None = None) -> list[Path]:
    """Return the template and the files it includes, recursively (each once)."""
    case_dir = (case_dir or template_path.resolve().parent).resolve()
    seen: list[Path] = []
    aliases: dict[str, str] = {}
    pending = [template_path.resolve()]
    while pending and len(seen) < 50:
        path = pending.pop(0)
        if path in seen or not path.is_file() or path.stat().st_size > _SCAN_MAX_BYTES:
            continue
        seen.append(path)
        for keyword, records in _keyword_records(path.read_text(errors="replace")):
            if keyword == "PATHS":
                aliases.update(_path_aliases(records, template_path, case_dir))
            elif keyword == "INCLUDE" and records:
                target = _include_target(records[0], path, case_dir, aliases)
                if isinstance(target, Path):
                    pending.append(target)
    return seen


def _path_aliases(records: list[str], template_path: Path, case_dir: Path) -> dict[str, str]:
    """Return a ``PATHS`` keyword's aliases (``'FLOC' 'dir'``) to their folders.

    ``${case_dir}`` is substituted; a relative folder is taken from the template's (OPM
    takes it from the rendered deck's, the member's folder, to which the template's
    own files are copied).
    """
    aliases = {}
    for record in records:
        fields = _expand(record)
        if len(fields) >= 2:
            folder = Path(fields[1].replace("${case_dir}", str(case_dir)))
            if not folder.is_absolute() and "$" not in str(folder):
                folder = template_path.resolve().parent / folder
            aliases[fields[0]] = str(folder)
    return aliases


def _include_target(
    record: str, including: Path, case_dir: Path, aliases: Mapping[str, str] | None = None
) -> Path | str | None:
    """Return an INCLUDE record's file, or its name if it holds a template expression.

    ``$ALIAS`` is a ``PATHS`` alias (``aliases``). A relative path is taken from the
    including file's folder; ``None`` if empty.
    """
    match = _INCLUDE_PATH.search(record)
    if match is None:
        return None
    name = (match.group(1) or match.group(2)).replace("${case_dir}", str(case_dir))
    for alias, folder in (aliases or {}).items():
        name = re.sub(rf"\${re.escape(alias)}(?![A-Za-z0-9_])", lambda _, f=folder: f, name)
    if "$" in name:
        return name  # a template expression, or a PATHS alias not defined
    target = Path(name)
    return target if target.is_absolute() else including.parent / target


def _keyword_records(
    text: str, single_record: frozenset[str] = frozenset()
) -> list[tuple[str, list[str]]]:
    """Split deck text into keywords and their ``/``-terminated records (comments dropped).

    Linear in the text: each line is searched for its record's ``/`` once, with the quote
    state carried over from the record's earlier lines, and a record's lines are joined
    once. (It rescanned the record so far at every line: quadratic, many minutes for one
    property array of 60,000 lines, such as Brugge's NTG; 2026-10-06.)

    The keywords in ``single_record`` (e.g. ``RPTRST``) take one record whose items may
    stand alone on their lines (``BASIC=2`` then ``FLOWS``): until that record ends, a bare
    word is an item, not a new keyword.
    """
    found: list[tuple[str, list[str]]] = []
    records: list[str] | None = None
    parts: list[str] = []
    quote = ""
    current = ""
    for raw in text.splitlines():
        line = raw.split("--", 1)[0].strip()
        if not line or line.startswith(("%", "<%", "##")):
            continue
        keyword = _KEYWORD.match(line)
        # A ruler after a section keyword, but not in a TITLE's free text ("GRID test") or
        # in a record continued onto this line.
        if keyword is None and not parts and not (current == "TITLE" and records == []):
            keyword = _SECTION_RULER.match(line)
        if keyword and not (current in single_record and records == []):
            current = keyword.group(1)
            # A keyword starts afresh (some, such as TITLE, take a record with no "/").
            records, parts, quote = [], [], ""
            found.append((keyword.group(1), records))
            continue
        if records is None:
            continue
        end, quote = _terminator(line, quote)
        if end is None:
            parts.append(line)
            continue
        body = " ".join([*parts, line[:end]]).strip()
        parts, quote = [], ""
        if body:
            records.append(body)
        else:
            records = None  # a lone "/" ends the keyword's records
    return found


def keyword_records(text: str) -> list[tuple[str, list[str]]]:
    """Return deck text's keywords, each with its ``/``-terminated records, in order.

    The text is read as it stands: includes are not followed (:func:`deck_keywords` follows
    them) and Mako lines are skipped. Comments are dropped, and as in Eclipse and OPM Flow
    the text after a record's ``/`` on its line is a comment too (``PERMZ 1 / PERMY 2 /``
    on one line is one record). Keywords are upper case; records are as written.
    """
    return _keyword_records(text)


def include_path(record: str) -> str | None:
    """Return an ``INCLUDE`` record's path as written (quotes removed), or None if empty.

    Template expressions (``${case_dir}/grid.inc``) are kept as they are.
    """
    match = _INCLUDE_PATH.search(record)
    if match is None:
        return None
    return (match.group(1) or match.group(2)).strip()


def _terminator(line: str, quote: str) -> tuple[int | None, str]:
    """Return where a record's ``/`` is in ``line``, and the quote open at its end.

    The ``/`` is not one inside quotes; ``None`` if there is none. ``quote`` is the quote
    open at the line's start (from the record's earlier lines).
    """
    for index, char in enumerate(line):
        if quote:
            quote = "" if char == quote else quote
        elif char in "'\"":
            quote = char
        elif char == "/":
            return index, ""
    return None, quote


def _expand(record: str) -> list[str]:
    """Return a record's fields, with ``n*`` defaults expanded and quotes removed."""
    fields: list[str] = []
    for token in record.split():
        repeat = re.fullmatch(r"(\d+)\*(.*)", token)
        if repeat:
            fields += [repeat.group(2) or "*"] * int(repeat.group(1))
        else:
            fields.append(token.strip("'\""))
    return fields


def _completion(
    fields: list[str], heads: Mapping[str, tuple[int, int]]
) -> tuple[str, list[list[int]]] | None:
    """Return a COMPDAT record's well and cells, or ``None`` if it cannot be read."""
    if len(fields) < 5:
        return None
    name, i, j, k1, k2 = fields[:5]
    head = heads.get(name)
    # A defaulted or 0 location is the well head's (WELSPECS).
    if i in ("*", "0") and head:
        i = str(head[0])
    if j in ("*", "0") and head:
        j = str(head[1])
    if not all(v.isdigit() for v in (i, j, k1, k2)) or int(k2) < int(k1):
        return None
    return name, [[int(i), int(j), k] for k in range(int(k1), int(k2) + 1)]


_MONTHS = {m: i for i, m in enumerate(
    ("JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"), 1
)}  # fmt: skip


def _date(fields: list[str]) -> "dt.date | None":
    """Return an Eclipse date record (``31 JAN 1970``, ``'JLY'`` as July), or ``None``."""
    if len(fields) < 3 or not fields[0].isdigit() or not fields[2].isdigit():
        return None
    month = fields[1].upper().replace("JLY", "JUL")
    if month not in _MONTHS:
        return None
    try:
        return dt.date(int(fields[2]), _MONTHS[month], int(fields[0]))
    except ValueError:
        return None


def schedule_summary(template_path: Path, case_dir: Path | None = None) -> dict[str, object]:
    """Return a deck's schedule as an optimization needs it, read from its text.

    Reads ``START``, ``DATES`` (report dates, in order), ``WELSPECS`` (each well's preferred
    phase) and the well controls ``WCONHIST``/``WCONPROD`` (producers, with their largest
    oil, water and liquid rates) and ``WCONINJE``/``WCONINJH`` (injectors, their type and
    largest rate), from the template and the files it includes (``${case_dir}`` as in
    :func:`deck_layout`). Nothing is rendered.

    Returns
    -------
    dict
        ``start``: the ``START`` date or ``None``; ``dates``: the ``DATES`` in order;
        ``producers``/``injectors``: well to ``{"phase", "max_oil_rate", "max_water_rate",
        "max_liquid_rate"}`` or ``{"type", "max_rate"}``; ``shut``: wells last set ``SHUT`` or
        never producing or injecting;
        ``tstep``: whether ``TSTEP`` is used (its steps are not dates);
        ``unreadable``: wells whose control records hold template expressions (``${...}``,
        Mako lines), so their rates cannot be read.
    """
    start: dt.date | None = None
    dates: list[dt.date] = []
    phases: dict[str, str] = {}
    producers: dict[str, dict[str, float]] = {}
    injectors: dict[str, dict[str, object]] = {}
    status: dict[str, str] = {}
    unreadable: set[str] = set()
    tstep = False
    on_pressure: set[str] = set()

    def number(value: str) -> float:
        try:
            return float(value)
        except ValueError:
            return 0.0

    for path in _deck_files(template_path, case_dir):
        for keyword, records in _keyword_records(path.read_text(errors="replace")):
            if keyword == "START" and records and start is None:
                start = _date(_expand(records[0]))
            elif keyword == "DATES":
                dates += [d for d in (_date(_expand(r)) for r in records) if d is not None]
            elif keyword == "TSTEP":
                tstep = True
            elif keyword == "WELSPECS":
                for record in records:
                    fields = _expand(record)
                    if len(fields) >= 6:
                        phases[fields[0]] = fields[5].upper()
            elif keyword in ("WCONHIST", "WCONPROD"):
                for record in records:
                    fields = _expand(record)
                    if len(fields) < 3:
                        continue
                    name, open_ = fields[0], fields[1].upper()
                    status[name] = open_
                    if any("$" in f or "%" in f for f in fields[1:6]):
                        unreadable.add(name)
                    # WCONHIST: name status mode ORAT WRAT GRAT; WCONPROD: ... ORAT WRAT GRAT LRAT
                    oil, water = (number(fields[3]) if len(fields) > 3 else 0.0,
                                  number(fields[4]) if len(fields) > 4 else 0.0)  # fmt: skip
                    best = producers.setdefault(name, {"max_oil_rate": 0.0, "max_water_rate": 0.0,
                                                       "max_liquid_rate": 0.0})  # fmt: skip
                    best["max_oil_rate"] = max(best["max_oil_rate"], oil)
                    best["max_water_rate"] = max(best["max_water_rate"], water)
                    # WCONPROD's LRAT: a producer on a liquid-rate target (as the deck writer's
                    # synthetic cases) produces with no oil or water rate given.
                    liquid = number(fields[6]) if keyword == "WCONPROD" and len(fields) > 6 else 0.0
                    best["max_liquid_rate"] = max(best["max_liquid_rate"], oil + water, liquid)
                    if keyword == "WCONPROD" and len(fields) > 2 and fields[2].upper() == "BHP":
                        on_pressure.add(name)  # produces at its pressure, with no rate given
            elif keyword in ("WCONINJE", "WCONINJH"):
                for record in records:
                    fields = _expand(record)
                    if len(fields) < 3:
                        continue
                    name = fields[0]
                    status[name] = fields[2].upper()
                    if any("$" in f or "%" in f for f in fields[1:6]):
                        unreadable.add(name)
                    rate = number(fields[4]) if keyword == "WCONINJE" and len(fields) > 4 else (
                        number(fields[3]) if len(fields) > 3 else 0.0
                    )  # fmt: skip
                    best_i = injectors.setdefault(
                        name, {"type": fields[1].upper(), "max_rate": 0.0}
                    )
                    best_i["max_rate"] = max(float(cast(float, best_i["max_rate"])), rate)
    for name, info in producers.items():
        info_any = cast(dict[str, object], info)
        info_any["phase"] = phases.get(name, "OIL")
    # Shut: last set SHUT, or never producing or injecting (e.g. a completion shut in
    # COMPDAT, which the history's OPEN rows do not override).
    idle = {n for n, i in producers.items() if i["max_liquid_rate"] <= 0.0 and n not in on_pressure}
    idle |= {n for n, i in injectors.items() if float(cast(float, i["max_rate"])) <= 0.0}
    shut = sorted({n for n, s in status.items() if s == "SHUT"} | idle)
    return {
        "start": start, "dates": dates, "producers": producers, "injectors": injectors,
        "shut": shut, "tstep": tstep, "unreadable": sorted(unreadable),
    }  # fmt: skip


def case_folder(options: Mapping[str, object]) -> Path | None:
    """Return a deck simulator's ``case_dir`` option, if set (else ``None``: the template's)."""
    value = options.get("case_dir")
    return Path(str(value)) if value else None


# ---------------------------------------------------------------------- report and restart dates

#: The sections of a deck, in order (a keyword alone on its line starts each).
_SECTIONS = frozenset(
    {"RUNSPEC", "GRID", "EDIT", "PROPS", "REGIONS", "SOLUTION", "SUMMARY", "SCHEDULE"}
)
#: The sections whose keywords can change the report or restart times.
_TIMED_SECTIONS = frozenset({"SOLUTION", "SUMMARY", "SCHEDULE"})
#: The report keywords: one record, whose items may stand alone on their lines.
_REPORT_KEYWORDS = frozenset({"RPTRST", "RPTSCHED", "RPTSOL"})
#: Keywords that make the text's schedule differ from the run's (not followed here).
_UNFOLLOWED = frozenset({"SKIP", "SKIP100", "SKIP300", "ENDSKIP", "RESTART"})
#: A Mako line: control (``% for``), a block (``<% ... %>``) or a tag (``<%include/>``).
_MAKO_LINE = re.compile(r"^\s*(%(?!%)|<%|</%)")
_SECTION_LINE = re.compile(
    r"^\s*(RUNSPEC|GRID|EDIT|PROPS|REGIONS|SOLUTION|SUMMARY|SCHEDULE)\b[^\S\n]+\S.*$",
    re.MULTILINE,
)
_CLOCK = re.compile(r"(\d\d?):(\d\d)(?::(\d\d)(?:\.\d+)?)?")


@dataclass(frozen=True)
class RestartSchedule:
    """A deck's report times and the report times it writes restarts at.

    Read from the deck's text by :func:`restart_schedule`, as OPM Flow decides them
    (opm-common ``ScheduleState::rst_file``, ``RSTConfig``). It is advisory: the run's own
    restart files, each dated in its header, are what a reader matches, and ``unresolved``
    says why the text alone may not tell.

    Attributes
    ----------
    report_times
        ``START`` (report step 0), then each ``DATES`` record and ``TSTEP`` step, in order.
    restart_steps
        The report steps that write a restart, ascending.
    unresolved
        Why the schedule may differ from what the deck runs (template expressions, Mako
        lines, unread includes, settings not followed); empty when the text tells it all.
    """

    report_times: tuple[dt.datetime, ...]
    restart_steps: tuple[int, ...]
    unresolved: tuple[str, ...]

    @property
    def restart_times(self) -> tuple[dt.datetime, ...]:
        """The report times of :attr:`restart_steps`."""
        return tuple(self.report_times[k] for k in self.restart_steps)

    def missing(self, dates: Sequence[dt.date]) -> list[dt.date]:
        """Return the ``dates`` on which no restart is written, in their order."""
        written = {t.date() for t in self.restart_times}
        return [d for d in dates if d not in written]


@dataclass
class _RestartControl:
    """OPM's ``RSTConfig``: BASIC, FREQ, and ``write``.

    ``write`` is OPM's ``write_rst_file``: ``False`` for BASIC 0, ``True`` for 1 and 2,
    and ``None`` (decided per report step from BASIC and FREQ) for 3 and over, or before
    any control.
    """

    basic: int | None = None
    freq: int | None = None
    write: bool | None = None

    def update(self, basic: int | None, freq: int | None) -> None:
        """Set BASIC (and so ``write``) and FREQ where given; FREQ keeps its last value."""
        if basic is not None:
            self.basic = basic
            self.write = False if basic == 0 else True if basic in (1, 2) else None
        if freq is not None:
            self.freq = freq


def _mnemonic(record: str, name: str) -> int | None:
    match = re.search(rf"\b{name}\s*=\s*(-?\d+)", record, re.IGNORECASE)
    return int(match.group(1)) if match else None


def _clock(fields: list[str]) -> dt.time | None:
    """Return a date record's time item (the 4th; midnight if none), ``None`` if invalid."""
    if len(fields) < 4:
        return dt.time()
    match = _CLOCK.fullmatch(fields[3])
    if match is None:
        return None
    hour, minute, second = (int(g) if g else 0 for g in match.groups())
    try:
        return dt.time(hour, minute, second)
    except ValueError:  # e.g. 24:00:00
        return None


def _ordered_records(
    template_path: Path,
    case_dir: Path | None,
    note: Callable[[str], None],
    unread: Callable[[str], None] | None = None,
    blocked: Callable[[set[str]], None] | None = None,
) -> list[tuple[str, list[str], bool]]:
    """Return the deck's keywords and records in deck order, includes expanded in place.

    Each item is ``(keyword, records, mako)``, with ``mako`` true when the keyword's file
    has Mako lines from ``SOLUTION`` on (they may add or drop records). Reading stops at
    ``END``. What cannot be read from ``SOLUTION`` on is noted; earlier sections (the
    grid, the properties) cannot change the report times. ``unread`` hears what cannot be
    read in any section (an include not read or with a templated path, a line that is a
    template expression outside a Mako block or a Mako include), for a keyword's absence to
    mean anything (e.g. no ``RTEMP``); ``blocked`` hears the keywords inside each file's
    Mako blocks (:func:`_keywords_in_mako_blocks`), whose records depend on the rendering.
    """

    def lost(message: str) -> None:
        if unread is not None:
            unread(message)

    case_dir = (case_dir or template_path.resolve().parent).resolve()
    out: list[tuple[str, list[str], bool]] = []
    aliases: dict[str, str] = {}
    section = ""
    ended = False
    files = 0

    def timed() -> bool:
        return section in _TIMED_SECTIONS

    def walk(path: Path, depth: int, conditional: bool = False) -> None:
        # ``conditional``: the file is included from inside a Mako block, so whether its
        # records render at all depends on the rendering.
        nonlocal section, ended, files
        files += 1
        if files > 200:
            note("more than 200 included files; the rest are not read")
            lost("more than 200 included files")
            return
        if depth > 10:
            if timed():
                note(f"includes nested deeper than 10 at {path.name}")
            lost(f"includes nested deeper than 10 at {path.name}")
            return
        if not path.is_file():
            if timed():
                note(f"included file {path} not found")
            lost(f"included file {path} not found")
            return
        if path.stat().st_size > _SCAN_MAX_BYTES:
            if timed():
                note(f"{path.name} is over {_SCAN_MAX_BYTES // 1_000_000} MB and is not read")
            lost(f"{path.name} is not read (too large)")
            return
        # Eclipse ignores text after a section keyword ("SCHEDULE  ====="); the keyword
        # alone is what the record reader recognises.
        text = _SECTION_LINE.sub(r"\1", path.read_text(errors="replace"))
        # Mako lines from where the timed sections start in this file (or all of it).
        lines = text.splitlines()
        first = 0 if timed() else next(
            (i for i, line in enumerate(lines)
             if line.split("--", 1)[0].strip() in _TIMED_SECTIONS),
            len(lines),
        )  # fmt: skip
        mako = any(_MAKO_LINE.match(line) for line in lines[first:])
        # A template expression alone on its line may render to whole keywords (DATES,
        # RPTRST) that the text does not show; within a record it is noted by its keyword.
        if any(line.split("--", 1)[0].strip().startswith("${") for line in lines[first:]):
            note(f"a line of {path.name} is a template expression (it may render keywords)")
        # A Mako include, or an expression line outside a Mako block, may render any
        # keyword; inside a block (a loop over a state's values) it renders records.
        if _free_expressions(lines):
            lost(f"a line of {path.name} is a template expression or a Mako include")
        in_blocks = _keywords_in_mako_blocks(text)
        items = list(_keyword_records(text, _REPORT_KEYWORDS))
        if blocked is not None:
            blocked(in_blocks)
            if conditional:
                blocked({keyword for keyword, _ in items})
        for keyword, records in items:
            if ended:
                return
            if keyword == "END":
                ended = True
                return
            if keyword in _SECTIONS:
                section = keyword
            if keyword == "PATHS":
                aliases.update(_path_aliases(records, template_path, case_dir))
            if keyword == "INCLUDE" and records:
                if mako and timed():
                    note("Mako lines around INCLUDE")
                target = _include_target(records[0], path, case_dir, aliases)
                if isinstance(target, str):
                    if timed():
                        note(f"INCLUDE of a template expression or undefined PATHS alias: {target}")
                    lost(f"INCLUDE {target}")
                elif target is not None:
                    walk(target, depth + 1, conditional or "INCLUDE" in in_blocks)
                continue
            # A conditionally included file's records may not render at all.
            out.append((keyword, records, (mako or conditional) and timed()))

    walk(template_path.resolve(), 0)
    return out


def restart_schedule(template_path: Path, case_dir: Path | None = None) -> RestartSchedule:
    """Return the report times of a deck and those it writes restarts at, from its text.

    Follows OPM Flow (opm-common ``RSTConfig``, ``ScheduleState::rst_file``):

    - ``RPTRST`` (mnemonics ``BASIC=``, ``FREQ=``, or the integer form, whose items 1 and
      6 are BASIC and FREQ) and ``RPTSCHED`` ``RESTART=n`` (BASIC ``min(2, n)``, FREQ 1,
      unless BASIC is over 2) set the control; FREQ keeps its last value when not given.
      ``RPTSCHED NOTHING`` clears it.
    - Report step 0 (``START``) has a restart when the control in force before the first
      report time (``SOLUTION``'s ``RPTRST`` and the schedule's first block) has BASIC 1
      or 2, or ``RPTSOL`` ``RESTART`` > 1. (Measured on 5Spots: OPM Flow writes ``X0000``
      for ``RPTRST BASIC=2`` at the top of the schedule; the ``SOLUTION`` case follows
      opm-common's ``RSTConfig::first``, not a run.) BASIC 3 to 5 write none there (OPM's
      ``must_write_rst_file(0)`` reads only ``write_rst_file``); ``SAVE`` there is noted.
    - Report step k ≥ 1, at the k-th report time, is decided by the control in force
      when that time is reached, i.e. after the keywords before it (opm-common's
      ``Schedule::must_write_rst_file`` reads ``snapshots[k - 1].rst_config()``): BASIC 0
      (or none) never; 1 and 2 always; 3 when ``k % FREQ == 0``; 4 (5) at the first
      report time of a year (month) at least FREQ years (months) after the last restart;
      ``SAVE`` in the block that time starts, always (not a restart event: BASIC 4 and
      5 count from the last restart the control wrote). BASIC 1 is taken as OPM takes
      it, a restart every time (Eclipse overwrites them unless unified).

    Reading stops at ``END``. ``${case_dir}`` is ``case_dir``, as in :func:`deck_layout`.
    Nothing is rendered.

    Returns
    -------
    RestartSchedule
        The report times, the restart steps, and what the text cannot tell.
    """
    unresolved: list[str] = []

    def note(message: str) -> None:
        if message not in unresolved:
            unresolved.append(message)

    events = _ordered_records(template_path, case_dir, note)
    control = _RestartControl()
    times: list[dt.datetime] = []
    steps: list[int] = []  # restart events (OPM's restart_output)
    saved: set[int] = set()  # steps SAVE writes
    initial = False

    def decide(k: int) -> None:
        """Decide report step k's restart from the control in force before its time."""
        if k == 0:
            if control.write is True or (initial and control.write is not False):
                steps.append(0)
            elif initial:
                note("RPTSOL RESTART and RPTRST BASIC=0 before the first report time")
            return
        basic, freq = control.basic or 0, control.freq or 1
        if control.write is True:
            write = True
        elif control.write is False or basic == 0:
            write = False
        else:
            last = times[steps[-1]] if steps else times[0]
            now, before = times[k], times[k - 1]
            years = now.year - last.year
            months = 12 * years + now.month - last.month
            if basic == 3:
                write = k % freq == 0
            elif basic == 4:
                write = now.year > before.year and years >= freq
            elif basic == 5:
                write = (now.year, now.month) > (before.year, before.month) and months >= freq
            else:
                note(f"RPTRST BASIC={basic} is not supported (OPM refuses it)")
                write = False
            if basic in (4, 5) and freq > 1:
                note(f"RPTRST BASIC={basic} with FREQ={freq}: the restart dates are checked "
                     "when the run is read")  # fmt: skip
        if write:
            steps.append(k)

    def advance(time: dt.datetime) -> None:
        """Add the next report time and decide its restart (and the start's, first)."""
        if len(times) == 1:
            decide(0)
        times.append(time)
        decide(len(times) - 1)

    for keyword, records, mako in events:
        if keyword in _UNFOLLOWED:
            note(f"{keyword} is not followed")
            continue
        if keyword not in ("START", "DATES", "TSTEP", "SAVE") and keyword not in _REPORT_KEYWORDS:
            continue
        if mako and keyword != "START":
            note(f"Mako lines around {keyword}")
        if any("$" in r for r in records):
            note(f"{keyword} holds template expressions")
        if keyword == "START" and records and not times:
            fields = _expand(records[0])
            date, clock = _date(fields), _clock(fields)
            if date is None or clock is None:
                note("START is not a date")
                break
            times.append(dt.datetime.combine(date, clock))
            continue
        if not times and keyword in ("DATES", "TSTEP"):
            note("no START date before the schedule (OPM's default is 1 JAN 1983)")
            break
        if keyword in _REPORT_KEYWORDS:
            record = " ".join(records)
            fields = _expand(record)
            integer_form = bool(fields) and fields[0].lstrip("-").isdigit()
            if fields and fields[0] == "*":
                note(f"{keyword} with a defaulted first item is not read")
            if keyword == "RPTSOL":
                restart = _mnemonic(record, "RESTART")
                if integer_form:
                    note("RPTSOL in its integer form is not read")
                initial = initial or (restart is not None and restart > 1)
            elif keyword == "RPTSCHED":
                if re.search(r"\bNOTHING\b", record, re.IGNORECASE):
                    control.basic, control.write = None, None
                restart = _mnemonic(record, "RESTART")
                if restart is not None and (control.basic or 0) <= 2:
                    control.update(min(2, restart), 1)
                elif integer_form:
                    note("RPTSCHED in its integer form is not read")
            elif integer_form:
                freq = fields[5] if len(fields) > 5 and fields[5].isdigit() else None
                control.update(int(fields[0]), int(freq) if freq else None)
            else:
                control.update(_mnemonic(record, "BASIC"), _mnemonic(record, "FREQ"))
        elif keyword == "SAVE" and times:
            # OPM's updateSAVE marks the block it is in: a restart at that block's report
            # time (write_rst_file(k) or snapshots[k].save()), not a restart event, so
            # BASIC 4 and 5 do not count from it.
            saved.add(len(times) - 1)
        elif keyword == "DATES":
            for record in records:
                fields = _expand(record)
                date, clock = _date(fields), _clock(fields)
                if date is None or clock is None:
                    note(f"DATES record {record!r} is not a date")
                    continue
                advance(dt.datetime.combine(date, clock))
        elif keyword == "TSTEP":
            for value in _expand(" ".join(records)):
                try:
                    days = float(value)
                except ValueError:
                    note(f"TSTEP value {value!r} is not a number")
                    continue
                advance(times[-1] + dt.timedelta(days=days))
    if len(times) == 1:
        decide(0)
    if len(times) < 2 and not unresolved:
        note("no report times (DATES or TSTEP) found")
    return RestartSchedule(tuple(times), tuple(sorted({*steps, *saved})), tuple(unresolved))


def missing_summary_wells(
    template_path: Path, keyword: str, wells: Sequence[str], case_dir: Path | None = None
) -> list[str] | None:
    """Return the ``wells`` whose ``keyword`` vector the deck's SUMMARY does not request.

    Read from the template's text, as :func:`restart_schedule` reads it (includes under
    ``case_dir``), unrendered. A well keyword with no well (``WBHP`` then ``/``) asks for
    every well; quoted names may be patterns (``'PRO*'``). ``None`` when the text cannot
    tell: the keyword inside a Mako block or with a template expression, ``ALL`` (its set
    is the simulator's), a well list (``'*NAME'``), a record that does not split, a
    lower-case keyword, or anything left unread while the keyword is absent. A keyword
    with no well (``KEYWORD`` then ``/``) counts as every well only in the template's own
    text: one in an included file stays unknown, since the reader's empty record could
    also be one it dropped (an open quote).
    """
    unresolved: list[str] = []
    unread: list[str] = []
    in_mako_blocks: set[str] = set()
    events = _ordered_records(
        template_path, case_dir, unresolved.append, unread.append, in_mako_blocks.update
    )
    section, requested, unknown = "", [], False
    text = template_path.read_text(errors="replace")
    every_well = re.compile(
        rf"^\s*{re.escape(keyword)}\s*(--[^\n]*)?\n(\s*--[^\n]*\n)*\s*/",
        re.MULTILINE | re.IGNORECASE,
    )
    for raw, records, _ in events:
        name = raw.upper()
        if name in _SECTIONS:
            section = name
        if section != "SUMMARY":
            continue
        if name == "ALL":
            unknown = True
        if name != keyword.upper():
            # A keyword the reader did not take as one (lower case) lands in the previous
            # keyword's records, or SUMMARY's own when it comes first: the text cannot tell.
            if any(re.match(rf"\s*{re.escape(keyword)}\b", r, re.IGNORECASE) for r in records):
                unknown = True
            continue
        record = " ".join(records)
        if raw in in_mako_blocks or "$" in record:
            unknown = True
            continue
        try:
            # Quoted names may hold spaces ('PRO 1'); a record that does not split is
            # not known, and never counts as a missing well.
            names = [v for v in shlex.split(record, posix=True) if v != "/"]
        except ValueError:
            unknown = True
            continue
        if any(n.startswith("*") for n in names):
            unknown = True  # a well list ('*NAME', WLIST): its wells are the schedule's
            continue
        if not names and not every_well.search(text):
            # The reader drops a record it cannot close (an open quote), which then reads as
            # "every well": only the template's own `KEYWORD` + `/` is taken so.
            unknown = True
            continue
        requested.append(names or ["*"])  # no well: every well
    patterns = [n for names in requested for n in names]
    missing = [w for w in wells if not any(_well_matches(w, n) for n in patterns)]
    if missing and (unknown or unresolved or unread):
        return None
    return missing


def _well_matches(well: str, pattern: str) -> bool:
    """Whether a SUMMARY well name (Eclipse's trailing ``*`` patterns) names ``well``."""
    return fnmatch.fnmatchcase(well.upper(), pattern.upper())


@dataclass(frozen=True)
class DeckContents:
    """What a deck's text says about its run's output files, unrendered.

    Attributes
    ----------
    phases
        The phases RUNSPEC names (``oil``, ``water``, ``gas``), or ``None`` when its text
        cannot tell (Mako, template expressions or includes there, or no phase keyword).
    grid
        ``(nx, ny, nz)`` from ``DIMENS``/``SPECGRID``, or ``None`` (none, or a templated
        RUNSPEC).
    densities
        Whether an ``RPTRST`` asks for ``DEN`` (the phases' densities in the restarts);
        ``None`` when the text cannot tell (the integer form, template expressions, Mako,
        or what :func:`restart_schedule` notes).
    pore_volumes
        Whether an ``RPTRST`` asks for ``RPORV`` (the cells' pore volumes in the restarts);
        ``None`` as for ``densities``.
    unified, formatted
        ``UNIFOUT`` and ``FMTOUT`` in RUNSPEC; ``None`` when RUNSPEC is templated (Mako,
        expressions, includes).
    basic_one
        Every restart control writes ``BASIC=1`` (``RPTRST BASIC=1``, its integer form,
        or ``RPTSCHED RESTART=1``): Eclipse then keeps only the last of separate restarts.
        (With other controls too, which one is in force at a vintage is left to the
        restart check.) ``None`` when an RPTRST is templated (inside a Mako block) and the
        literal others are all ``BASIC=1``: whether only it holds is unknown.
    temperature
        The reservoir temperature in °C: ``RTEMPVD``'s (one temperature) or ``RTEMP``'s
        (converted from °F in a FIELD deck); in a CO2STORE deck with neither, OPM Flow's
        default of 60 °F (:data:`OPM_DEFAULT_TEMPERATURE_C`); else ``None`` (none in a
        black-oil deck, defaulted, templated, an unread include, or a templated RUNSPEC,
        whose unit system is unknown).
    temperature_varies
        The deck varies the temperature: ``RTEMPVD`` or ``TEMPVD`` with different
        temperatures (one temperature throughout is ``temperature``), or a thermal run
        (``THERMAL``, ``TEMPI``).
    temperature_tables
        The last ``RTEMPVD``/``TEMPVD``'s tables, one per equilibration region (``EQLNUM``
        1, 2, ...), each ``((depth_m, temperature_c), ...)`` in increasing depth, converted
        from FIELD units (ft, °F) where the deck is in them; ``None``
        without one, or when one is templated, defaulted, its RUNSPEC (units) templated, or
        the deck in LAB units.
    thermal
        A thermal run: RUNSPEC's ``THERMAL`` or ``TEMP``, or ``TEMPI``. OPM writes each
        cell's ``TEMP`` to the restarts only then.
    co2store
        RUNSPEC's ``CO2STORE`` (OPM Flow's CO₂-brine model), or ``None`` when RUNSPEC is
        templated. With ``OIL`` and ``GAS`` (its older form) the brine is the oil phase.
    gaswat
        RUNSPEC's ``GASWAT`` (Eclipse 300's gas-water model), or ``None`` when RUNSPEC is
        templated.
    compositional
        RUNSPEC's ``COMPS`` (Eclipse 300's compositional model, whose oil and gas come
        from the components: its RUNSPEC often names ``WATER`` only), or ``None`` when
        RUNSPEC is templated.
    init
        GRID's ``INIT``, which asks for the INIT file (the deck simulators write
        it only then): ``None`` when the text cannot tell (``INIT`` only inside a
        Mako block, or a part of the deck unread).
    """

    phases: frozenset[str] | None
    grid: tuple[int, int, int] | None
    densities: bool | None
    pore_volumes: bool | None
    unified: bool | None
    formatted: bool | None
    basic_one: bool | None
    temperature: float | None = None
    temperature_varies: bool = False
    co2store: bool | None = False
    temperature_tables: tuple[tuple[tuple[float, float], ...], ...] | None = None
    thermal: bool = False
    init: bool | None = None
    compositional: bool | None = False
    gaswat: bool | None = False


#: An ``INIT`` line in any case, with or without a record's ``/`` (keywords are read in
#: upper case alone on their line): not certainly absent.
_INIT_LINE = re.compile(r"^\s*INIT\s*(/\s*)?(--.*)?$", re.IGNORECASE | re.MULTILINE)


def _init_spelled(template_path: Path, case_dir: Path | None) -> bool:
    """Whether the template or a file it includes has an ``INIT`` line of another spelling.

    Includes are followed at any depth, as the deck reader follows them (``PATHS`` aliases
    and ``${case_dir}`` included); unreadable files and template expressions are skipped.
    """
    root = (case_dir or template_path.resolve().parent).resolve()
    pending: list[Path] = [template_path.resolve()]
    seen: set[Path] = set()
    aliases: dict[str, str] = {}
    while pending:
        path = pending.pop()
        if path in seen or len(seen) > 200:
            continue
        seen.add(path)
        try:
            text = path.read_text(errors="replace")
        except OSError:
            continue
        if _INIT_LINE.search(text):
            return True
        for keyword, records in _keyword_records(text):
            if keyword == "PATHS":
                aliases.update(_path_aliases(records, template_path, root))
            elif keyword == "INCLUDE" and records:
                target = _include_target(records[0], path, root, aliases)
                if isinstance(target, Path) and target.is_file():
                    pending.append(target.resolve())
    return False


def _free_expressions(lines: Sequence[str]) -> bool:
    """Return whether a line may render keywords the text does not show.

    That is a template expression (``${...}``) outside every Mako control block, or a
    Mako include (``<%include``, ``<%namespace``).
    """
    depth = 0
    for line in lines:
        stripped = line.split("--", 1)[0].strip()
        if stripped.startswith("%") and not stripped.startswith("%%"):
            opener = re.match(r"%\s*([A-Za-z]+)", stripped)
            word = opener.group(1) if opener else ""
            if word in ("if", "for", "while", "try", "with"):
                depth += 1
            elif word.startswith("end"):
                depth = max(0, depth - 1)
            continue
        if stripped.startswith(("<%include", "<%namespace")):
            return True
        if depth == 0 and stripped.startswith("${"):
            return True
    return False


def _keywords_in_mako_blocks(text: str) -> set[str]:
    """Return the keywords a Mako control block (``% if``/``% for``) encloses.

    Their records depend on the rendering, whichever section they are in.
    """
    out: set[str] = set()
    depth = 0
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("%") and not stripped.startswith("%%"):
            opener = re.match(r"%\s*([A-Za-z]+)", stripped)
            word = opener.group(1) if opener else ""
            if word in ("if", "for", "while", "try", "with"):
                depth += 1
            elif word.startswith("end"):
                depth = max(0, depth - 1)
            continue
        if depth and stripped and not stripped.startswith("--"):
            # The first token, without a trailing record end or comment (``RTEMP/``).
            word = re.split(r"[\s/]", stripped.split("--", 1)[0].strip())[0].upper()
            if _KEYWORD.match(word):  # not a record's data (numbers, ${...})
                out.add(word)
    return out


#: OPM Flow's reservoir temperature without RTEMP or RTEMPVD, 60 °F in °C: measured on
#: OPM Flow 2026.04 from a METRIC CO2STORE deck's CO₂ densities (Span & Wagner's at
#: 15.555 °C in every plume cell).
OPM_DEFAULT_TEMPERATURE_C = (60.0 - 32.0) / 1.8


def _celsius(
    values: Mapping[str, float | None],
    templated: Mapping[str, bool],
    field_units: bool,
    default: float | None = None,
) -> float | None:
    """Return the reservoir temperature in °C: the depth table's when given, else RTEMP's.

    ``default`` when neither keyword appears; ``None`` when the one in force is templated.
    """
    if not any(templated.values()) and values["RTEMP"] is None and values["RTEMPVD"] is None:
        return default
    source = "RTEMPVD" if values["RTEMPVD"] is not None or templated["RTEMPVD"] else "RTEMP"
    value = values[source]
    if value is None or templated[source]:
        return None
    return (value - 32.0) / 1.8 if field_units else value


def deck_contents(template_path: Path, case_dir: Path | None = None) -> DeckContents:
    """Return what a deck template's text says about its output (see :class:`DeckContents`).

    Read as :func:`restart_schedule` reads it (includes under ``case_dir``); nothing is
    rendered.
    """
    unresolved: list[str] = []
    unread: list[str] = []
    in_mako_blocks: set[str] = set()
    events = _ordered_records(
        template_path, case_dir, unresolved.append, unread.append, in_mako_blocks.update
    )
    section = ""
    phases: set[str] = set()
    basics: set[int] = set()
    den = porv = unknown = unified = formatted = varies = uncertain_basic = False
    # RTEMP's value and the depth tables' (one temperature), each with whether templated:
    # OPM takes the depth table's when both are given (measured), whatever their order.
    rtemp: dict[str, float | None] = {"RTEMP": None, "RTEMPVD": None}
    templated_temperature = {"RTEMP": False, "RTEMPVD": False}
    field_units = co2store = thermal = lab_units = False
    init = init_templated = compositional = gaswat = False
    # The last RTEMPVD/TEMPVD's tables in the deck's units, or None when one is unreadable.
    raw_tables: list[list[tuple[float, float]]] | None = None
    for keyword, records, mako in events:
        if keyword in _SECTIONS:
            section = keyword
            continue
        record = " ".join(records)
        if section == "GRID" and keyword == "INIT":
            if keyword in in_mako_blocks:
                init_templated = True
            else:
                init = True
        if keyword == "TEMPI" or (section == "RUNSPEC" and keyword in ("THERMAL", "TEMP")):
            varies = thermal = True
        elif keyword in ("RTEMPVD", "TEMPVD") and records:
            raw_tables = _depth_tables(records, keyword in in_mako_blocks)
            # Depth-temperature pairs: one temperature when every pair agrees (OPM's own
            # CO2STORE decks set it so), else a varying one.
            try:
                pairs = [float(v) for v in _expand(record) if v != "/"]
            except ValueError:
                pairs = []  # a template expression or a default
            # Only a keyword inside a Mako block is templated: a file's Mako elsewhere (a
            # schedule's loop) does not change these records.
            if pairs and not len(pairs) % 2 and len(set(pairs[1::2])) > 1:
                varies = True  # whatever renders around it, it varies
            elif keyword in in_mako_blocks or not pairs or len(pairs) % 2:
                templated_temperature["RTEMPVD"] = True
            else:
                rtemp["RTEMPVD"] = pairs[1]
        elif keyword in ("RTEMP", "RTEMPA") and records:
            fields = _expand(record)
            try:
                rtemp["RTEMP"] = float(fields[0])
            except (IndexError, ValueError):
                templated_temperature["RTEMP"] = True  # a template expression or default
            if keyword in in_mako_blocks:
                templated_temperature["RTEMP"] = True
        if section == "RUNSPEC":
            field_units = field_units or keyword == "FIELD"
            lab_units = lab_units or keyword == "LAB"
            co2store = co2store or keyword == "CO2STORE"
            compositional = compositional or keyword == "COMPS"
            gaswat = gaswat or keyword == "GASWAT"
            if keyword in ("OIL", "WATER", "GAS"):
                phases.add(keyword.lower())
            unified = unified or keyword == "UNIFOUT"
            formatted = formatted or keyword == "FMTOUT"
        if keyword == "RPTRST" and keyword in in_mako_blocks:
            # Inside a Mako block, or in a file included from one: it may not render, so
            # what it asks for is not known, its BASIC= neither (so not "only BASIC=1").
            unknown = uncertain_basic = True
        elif keyword == "RPTRST":
            unknown = unknown or mako or "$" in record
            fields = _expand(record)
            # The integer form (its first item a number, or defaulted) has no mnemonics.
            if fields and (fields[0] == "*" or fields[0].lstrip("-").isdigit()):
                unknown = True
                if fields[0] != "*":
                    basics.add(int(fields[0]))
            else:
                den = den or re.search(r"\bDEN\b", record, re.IGNORECASE) is not None
                porv = porv or re.search(r"\bRPORV\b", record, re.IGNORECASE) is not None
                # ALLPROPS asks for every property, the densities among them in Eclipse;
                # whether OPM's is the same is not checked here, so the text cannot tell.
                unknown = unknown or re.search(r"\bALLPROPS\b", record, re.IGNORECASE) is not None
                if (basic := _mnemonic(record, "BASIC")) is not None:
                    basics.add(basic)
        elif keyword == "RPTSCHED" and (restart := _mnemonic(record, "RESTART")) is not None:
            basics.add(min(2, restart))
    text = template_path.read_text(errors="replace")
    runspec = re.search(r"^\s*RUNSPEC\b(.*?)^\s*GRID\b", text, re.MULTILINE | re.DOTALL)
    # RUNSPEC is known only from the template's own lines without Mako, expressions or
    # includes (a missing include is noted only from SOLUTION on).
    templated = runspec is None or any(
        _MAKO_LINE.match(line)
        or "${" in line
        or line.split("--", 1)[0].strip().upper().split()[:1] == ["INCLUDE"]
        for line in runspec.group(1).splitlines()
    )
    grid = deck_layout(template_path, case_dir)["grid"]
    return DeckContents(
        phases=frozenset(phases) if phases and not templated else None,
        # deck_layout takes the first DIMENS and ignores Mako: a templated RUNSPEC may
        # choose another.
        grid=tuple(grid) if isinstance(grid, list) and not templated else None,
        densities=True if den else (None if unknown or unresolved else False),
        pore_volumes=True if porv else (None if unknown or unresolved else False),
        unified=None if templated else unified,
        formatted=None if templated else formatted,
        # A templated RPTRST's BASIC is unknown: then only BASIC=1 may hold, or not.
        basic_one=(None if basics == {1} else False) if uncertain_basic else basics == {1},
        # A templated RUNSPEC may choose another unit system (°F or °C): unknown then.
        # Without RTEMP or RTEMPVD, OPM's CO2STORE takes its default for the CO₂'s density
        # (measured); a black-oil deck's temperature enters no fluid property, so it is
        # unknown there, as it is when any part of the deck could not be read.
        temperature=None
        if templated
        else _celsius(
            rtemp,
            templated_temperature,
            field_units,
            OPM_DEFAULT_TEMPERATURE_C if co2store and not unread and not varies else None,
        ),
        temperature_varies=varies,
        co2store=None if templated else co2store,
        # LAB's centimetres and other units are not converted: such tables are unread.
        temperature_tables=None
        if templated or raw_tables is None or lab_units
        else tuple(
            tuple((d * 0.3048, (t - 32.0) / 1.8) if field_units else (d, t) for d, t in table)
            for table in raw_tables
        ),
        thermal=thermal,
        init=True
        if init
        else (
            None if init_templated or unread or _init_spelled(template_path, case_dir) else False
        ),
        compositional=None if templated else compositional,
        gaswat=None if templated else gaswat,
    )


def _depth_tables(records: Sequence[str], in_mako: bool) -> list[list[tuple[float, float]]] | None:
    """Return ``RTEMPVD``/``TEMPVD``'s depth-temperature tables, one per record.

    ``None`` when a table is templated (inside a Mako block, or an expression), defaulted,
    has an odd count, or its depths do not increase (OPM refuses that too).
    """
    if in_mako:
        return None
    tables: list[list[tuple[float, float]]] = []
    for record in records:
        try:
            values = [float(v) for v in _expand(record) if v != "/"]
        except ValueError:
            return None  # a template expression or a default
        if not values or len(values) % 2:
            return None
        pairs = list(zip(values[0::2], values[1::2], strict=True))
        if any(b[0] <= a[0] for a, b in itertools.pairwise(pairs)):
            return None
        tables.append(pairs)
    return tables or None


# ------------------------------------------------------------------- stopping at a report date


#: Nested includes inlined at most this deep (a guard against an include of itself).
_MAX_INCLUDE_DEPTH = 10
#: Keywords that place report dates, or change what the cut reads, at a line's start.
_CUT_KEYWORDS = re.compile(
    r"^\s*(DATES|TSTEP|INCLUDE|SKIP|SKIP100|SKIP300|ENDSKIP|RESTART|END)\s*$", re.M
)


def inline_schedule_includes(
    text: str,
    resolve: Callable[[str], Path | None],
    until: dt.date | None = None,
) -> str:
    """Return ``text`` with each ``INCLUDE`` of its ``SCHEDULE`` section replaced by the file.

    An include is text inclusion, so the deck that results runs as the original does; the
    cut at a date (:func:`stop_deck_at`) then sees the included ``DATES`` and ``TSTEP``
    records (decks whose schedule lives in includes). ``resolve`` maps an
    ``INCLUDE`` record's path, as the rendered deck names it, to the local file, or ``None``
    when it cannot (a relative path, a template expression, a ``PATHS`` alias, a missing
    file): that include is left as it is. Nested includes are inlined too. Includes outside
    the ``SCHEDULE`` section are left alone, and with ``until`` none after a ``DATES`` record
    past that date (the cut drops them). Text after an ``INCLUDE`` record's ``/`` is a
    comment to the simulators, and goes with the record.

    Raises
    ------
    ValueError
        If includes nest more than ten deep, or an included file that holds report
        keywords is not UTF-8 text (one that holds none is left to the simulator).
    """
    out, _ = _inlined(text, resolve, until, in_schedule=False, depth=0)
    return out


def _inlined(
    text: str,
    resolve: Callable[[str], Path | None],
    until: dt.date | None,
    *,
    in_schedule: bool,
    depth: int,
) -> tuple[str, bool]:
    """Return :func:`inline_schedule_includes`'s text, and whether ``until`` was passed."""
    if depth > _MAX_INCLUDE_DEPTH:
        raise ValueError("INCLUDE files nest more than ten deep (an include of itself?)")
    out: list[str] = []
    section = "SCHEDULE" if in_schedule else ""
    record: list[str] = []  # the INCLUDE keyword's lines while its record is read
    parts: list[str] = []  # its record's text, comments removed, up to the "/"
    body: list[str] = []  # another record's text, for a DATES record's date
    keyword = ""
    in_record = False
    quote = ""
    passed = False
    for raw in text.splitlines(keepends=True):
        line = raw.split("--", 1)[0].strip()
        if record:
            record.append(raw)
            if not line:
                continue
            end, quote = _terminator(line, quote)
            if end is None:
                parts.append(line)
                continue
            match = _INCLUDE_PATH.search(" ".join([*parts, line[:end]]).strip())
            target = None if match is None else resolve(match.group(1) or match.group(2))
            if target is None:
                out.extend(record)
            else:
                try:
                    content: str | None = target.read_text(encoding="utf-8")
                except UnicodeDecodeError as error:
                    # Comments in another encoding (Latin-1, often): such an include is left
                    # to the simulator unless it holds report dates, which the cut needs.
                    text_of = target.read_bytes().decode("latin-1")
                    if _CUT_KEYWORDS.search(text_of):
                        raise ValueError(
                            f"the schedule include {target.name} is not UTF-8 text "
                            f"({error.reason} at byte {error.start}) and holds report "
                            "keywords: save it as UTF-8 to cut the deck at a date"
                        ) from error
                    content = None
                if content is None:
                    out.extend(record)
                else:
                    included, passed = _inlined(
                        content, resolve, until, in_schedule=True, depth=depth + 1
                    )
                    out.append(f"-- APEX: the INCLUDE of {target.name} inlined, to cut at a date\n")
                    out.append(included if included.endswith("\n") else included + "\n")
            record, parts, quote = [], [], ""
            continue
        out.append(raw)
        if not line:
            continue
        if not in_record:
            if keyword == "TITLE":
                keyword = ""  # TITLE's one line of free text, with no "/"
                continue
            match = _KEYWORD.match(line) or _SECTION_RULER.match(line)
            if match is not None:
                keyword = match.group(1)
                if keyword in _SECTIONS:
                    section = keyword
                if keyword == "INCLUDE" and section == "SCHEDULE" and not passed:
                    out.pop()
                    record = [raw]
                continue
        end, quote = _terminator(line, quote)
        in_record = end is None
        if keyword == "DATES" and section == "SCHEDULE" and until is not None:
            body.append(line if end is None else line[:end])
            if end is not None:
                date = _date(_expand(" ".join(body)))
                body = []
                passed = passed or (date is not None and date > until)
    if record:
        out.extend(record)  # an unterminated INCLUDE: left for the simulator to refuse
    return "".join(out), passed


def stop_deck_at(
    text: str,
    stop_at: dt.date,
    report_dates: Sequence[dt.date] | None = None,
    resolve: Callable[[str], Path | None] | None = None,
) -> str:
    """Return a rendered deck that ends at the report date ``stop_at``.

    The ``SCHEDULE`` section is cut after the ``DATES`` record of ``stop_at`` (its keyword
    closed), or after the ``TSTEP`` step reaching it (the record shortened to the steps up
    to it), and ``END`` follows: the run is the deck's own up to that date, the same report
    steps, nothing after it. Everything before the cut is kept byte for byte.

    The date must be one of the deck's report dates, read from the main text: ``START``,
    then each ``DATES`` record and ``TSTEP`` step (whole days). Includes in the
    ``SCHEDULE`` section are not opened here: a ``DATES`` record after one places itself
    (its date is absolute), and a ``TSTEP`` after one counts from that ``DATES`` record. A
    ``TSTEP`` after an include and before any later ``DATES`` is refused, unless
    ``report_dates`` (the deck's report dates as :func:`restart_schedule` reads them from
    the template, includes followed) is given: then the main text's report dates up to the
    cut must be those, which they are only when the includes hold no ``DATES`` or ``TSTEP``.

    With ``resolve``, the ``SCHEDULE`` section's includes are first inlined
    (:func:`inline_schedule_includes`), so a deck whose report dates live in includes is
    cut too; the text before the cut is then the deck's with those files in place of their
    ``INCLUDE`` records, which the simulator reads alike.

    Refused besides: ``SKIP``, ``SKIP100``, ``SKIP300``, ``ENDSKIP`` or ``RESTART`` before the
    cut (the main text's schedule is then not the run's), and a ``DATES`` record with a time
    of day on or before the date (a date names one report only at midnight).

    Raises
    ------
    ValueError
        As said, or if ``stop_at`` is not a report date after ``START`` in the main text.
    """
    if resolve is not None:
        text = inline_schedule_includes(text, resolve, stop_at)
    lines = text.splitlines(keepends=True)
    start: dt.date | None = None
    section = ""
    keyword = ""
    record: list[str] = []
    record_from = 0
    quote = ""
    now: dt.date | None = None
    included = False
    seen: list[dt.date] = []  # the main text's report dates, START first

    def checked(cut: str) -> str:
        # With the template's dates, the includes must have added none up to the cut.
        if report_dates is not None and list(report_dates[: len(seen)]) != seen:
            raise ValueError(
                "the deck's includes hold report dates (DATES or TSTEP) before "
                f"{stop_at.isoformat()}: its main text alone cannot place the cut"
            )
        return cut

    for index, raw in enumerate(lines):
        line = raw.split("--", 1)[0].strip()
        if not line:
            continue
        match = _KEYWORD.match(line) if not record else None
        if match is None and not record:
            ruler = re.match(
                r"^(RUNSPEC|GRID|EDIT|PROPS|REGIONS|SOLUTION|SUMMARY|SCHEDULE)\b", line
            )
            match = ruler
        if keyword == "TITLE" and not record and match is None:
            keyword = ""  # TITLE's one line of free text, with no "/"
            continue
        if match is None and not record and re.match(r"^(DATES|TSTEP)\s+\S", line):
            # Records on the keyword's own line: the reader takes the keyword alone on its
            # line, so this would be misread rather than cut.
            raise ValueError(f"{line.split()[0]} with its records on its own line: {line!r}")
        if match is not None:
            keyword = match.group(1)
            if keyword in _SECTIONS:
                section = keyword
                if keyword == "SCHEDULE":
                    now = start
            if keyword == "END":
                break
            if keyword in _UNFOLLOWED:
                raise ValueError(
                    f"{keyword} before {stop_at.isoformat()}: the deck's schedule text is not "
                    "what it runs, so it cannot be cut at a date"
                )
            continue
        if not record:
            record_from = index
        end, quote = _terminator(line, quote)
        if end is None:
            record.append(line)
            continue
        body = " ".join([*record, line[:end]]).strip()
        record, quote = [], ""
        if not body:
            continue  # a lone "/" closes a keyword's records
        fields = _expand(body)
        if keyword == "START":
            start = _date(fields)
            if start is not None:
                seen.append(start)
            continue
        if section != "SCHEDULE":
            continue
        if keyword == "INCLUDE":
            included = True
        elif keyword == "DATES":
            date = _date(fields)
            if date is None:
                raise ValueError(f"a DATES record the deck reader cannot read: {body!r}")
            if date <= stop_at and _clock(fields) != dt.time():
                raise ValueError(
                    f"a DATES record at a time of day ({body!r}): a date names a report only "
                    "at midnight"
                )
            now = date
            included = False  # the time is known again
            seen.append(date)
            if date == stop_at:
                # The record kept up to its "/" (a later record on the same line dropped),
                # its keyword closed, and the run ended there.
                raw = lines[index].split("--", 1)[0]
                upto = raw.index(line) + end + 1
                kept = "".join(lines[:index]) + raw[:upto] + "\n"
                return checked(kept + "/\n\nEND\n")
            if date > stop_at:
                break
        elif keyword == "TSTEP":
            if included and report_dates is None:
                raise ValueError(
                    "a TSTEP follows an INCLUDE in the SCHEDULE section: the time it starts "
                    "from is not known from the main text"
                )
            if now is None:
                raise ValueError("the deck has no START before its TSTEP")
            days: list[str] = []
            for value in fields:
                step = float(value)
                if step != int(step):
                    raise ValueError(f"TSTEP {value} is not whole days")
                now = now + dt.timedelta(days=int(step))
                days.append(value)
                seen.append(now)
                if now == stop_at:
                    shortened = " " + " ".join(days) + " /\n"
                    return checked("".join(lines[:record_from]) + shortened + "\nEND\n")
                if now > stop_at:
                    break
            if now > stop_at:
                break
    if start is None:
        raise ValueError("the deck has no START: its report dates cannot be placed")
    raise ValueError(
        f"{stop_at.isoformat()} is not a report date of the deck's main text (its DATES and "
        "TSTEP records after START)"
    )


def stopped_deck(
    text: str,
    member: MemberInput,
    report_dates: Sequence[dt.date] | None = None,
    resolve: Callable[[str], Path | None] | None = None,
) -> str:
    """Return ``text`` cut at ``member.stop_at`` (:func:`stop_deck_at`); unchanged without it.

    ``report_dates``: the template's report dates (:func:`template_report_dates`), which let
    a deck whose schedule includes come before a ``TSTEP`` be cut. ``resolve``: the local
    file of an ``INCLUDE`` path (:func:`include_resolver`), to inline schedule includes.

    Raises
    ------
    SimulationError
        If ``stop_at`` is not an ISO date or not a report date of the deck.
    """
    if member.stop_at is None:
        return text
    try:
        return stop_deck_at(text, dt.date.fromisoformat(member.stop_at), report_dates, resolve)
    except OSError as error:
        # Reading an include (a stale file handle, an I/O error) may not recur: retried.
        raise TransientSimulationError(
            f"member {member.member_id} cannot stop at {member.stop_at}: reading its "
            f"schedule includes failed: {error}"
        ) from error
    except ValueError as error:
        raise SimulationError(
            f"member {member.member_id} cannot stop at {member.stop_at}: {error}"
        ) from error


def include_resolver(
    template_path: Path, local_case_dir: Path | None, rendered_case_dir: Path | None = None
) -> Callable[[str], Path | None]:
    """Return a map from a rendered deck's absolute ``INCLUDE`` paths to local files.

    ``${case_dir}`` was rendered as ``rendered_case_dir`` (under a runner, the host's
    folder; default the local one), which is read here as ``local_case_dir`` (default the
    template's folder, as :func:`render_deck`'s). ``None``, so that the include is left to
    the simulator, for a relative path (the simulators take it from the deck's own folder,
    the member's work directory, which holds no copy here), a name with ``$`` (a ``PATHS``
    alias), a file that is not there, and under a runner an absolute path outside the
    rendered ``case_dir`` (the host's file may differ from the one here).
    """
    local = (local_case_dir or template_path.resolve().parent).resolve()
    # Here a path is resolved (symbolic links, ".."); a host's only normalised.
    rendered = local if rendered_case_dir is None else Path(rendered_case_dir).resolve()
    elsewhere = rendered != local
    if elsewhere:
        rendered = Path(os.path.normpath(str(rendered_case_dir)))

    def resolve(name: str) -> Path | None:
        if "$" in name:
            return None
        if not Path(name).is_absolute():
            return None
        path = Path(os.path.normpath(name)) if elsewhere else Path(name).resolve()
        if elsewhere:
            if not path.is_relative_to(rendered):
                return None
            path = local / path.relative_to(rendered)
        return path if path.is_file() else None

    return resolve


def template_report_dates(
    template_path: Path, case_dir: Path | None = None
) -> list[dt.date] | None:
    """Return a deck template's report dates, includes followed (:func:`restart_schedule`).

    ``None`` when the text alone may not give them (template expressions, Mako lines in the
    schedule, files not read: its ``unresolved``), or when it cannot be read.
    """
    try:
        schedule = restart_schedule(template_path, case_dir)
    except (OSError, ValueError):
        return None
    if schedule.unresolved:
        return None
    return [t.date() for t in schedule.report_times]
