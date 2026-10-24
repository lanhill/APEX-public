"""Turn a deck into an APEX deck template whose chosen properties come from the state.

Each chosen property's array (``PERMX``, ``PORO``, ...) is replaced by a Mako loop over the
member's state variable of the same name, lower case. With ``log`` the state is
``ln(value)`` and the deck gets ``exp(state)``. Cells that are 0 in the deck can be kept 0:
which ones is written into the template itself (run lengths), so rendering reads no file
and works wherever the deck runs (a remote host's ``case_dir`` too).

Included files are inlined when they hold what the template must see or change: a chosen
property, a section keyword, ``PATHS`` or an ``INCLUDE`` of their own (and so are the files
that include them). Every other file, a leaf of data, is copied beside the template and
included as ``${case_dir}/<path>``, which ``opm_flow`` and ``eclipse`` resolve. Everything
else is the deck's own text. Every chosen property must be replaced, or the template is
refused.

Rendering the template with the deck's own values (log ones as their logarithm) gives the
deck's values back, to the printed precision (``%.10g``).
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from apex.core.errors import ConfigError
from apex.simulation.conversion.deck_reader import (
    PROPERTY_SECTIONS,
    SECTIONS,
    Allowed,
    resolve_include,
    walk,
)
from apex.simulation.deck import _KEYWORD, _path_aliases, _terminator

HEADER = """<%!
import math


def _kept(runs):
    kept = []
    for flag, count in runs:
        kept.extend([flag] * count)
    return kept
%>
"""
#: Inlined files larger than this are noted: Mako compiles every line of the template.
INLINE_NOTE_BYTES = 20_000_000
#: Where files included from outside the deck's folder go, beside the template.
EXTERNAL = "external"


@dataclass(frozen=True)
class StateProperty:
    """A property the state carries: its state variable's name and its transform."""

    variable: str
    transform: str = "none"  # "none" or "log"
    #: Run lengths ``(flag, cells)`` over the cells: 1 where the state is written, 0 where the
    #: deck's 0 is kept (inactive cells, barriers). ``None``: every cell from the state.
    kept: tuple[tuple[int, int], ...] | None = None


@dataclass
class Template:
    """A deck template: its text, the files it includes (relative path to source) and notes."""

    text: str
    files: dict[str, Path] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


def write_template(
    deck: Path, chosen: Mapping[str, StateProperty], allowed: Allowed = None
) -> Template:
    """Return ``deck`` as a template in which the ``chosen`` properties are the state.

    Raises
    ------
    ConfigError
        If a chosen property is not set as an array in the deck.
    """
    deck = deck.resolve()
    parents: dict[Path, set[Path]] = {}
    marked: set[Path] = set()
    found: set[str] = set()
    aliases: dict[str, str] = {}
    for entry in walk(deck, allowed):  # every include checked before anything is copied
        here = entry.file.resolve()
        if entry.keyword == "PATHS":
            aliases.update(_path_aliases(entry.records, deck, deck.parent))
            marked.add(here)
        elif entry.keyword in ("INCLUDE", "INCLUDE?", "INCLUDE>"):
            marked.add(here)
            if entry.keyword == "INCLUDE":
                parents.setdefault(Path(entry.records[0]).resolve(), set()).add(here)
        elif entry.keyword in SECTIONS:
            marked.add(here)
        elif entry.section in PROPERTY_SECTIONS and entry.keyword in chosen:
            found.add(entry.keyword)
            marked.add(here)
    missing = sorted(set(chosen) - found)
    if missing:
        raise ConfigError(f"not set as an array in the deck: {', '.join(missing)}")
    inline: set[Path] = set()
    pending = list(marked)
    while pending:
        path = pending.pop()
        if path in inline or path == deck:
            continue
        inline.add(path)
        pending.extend(parents.get(path, ()))
    writer = _Writer(deck, chosen, inline, aliases)
    body = writer.process(deck)
    unreplaced = sorted(set(chosen) - writer.replaced)
    if unreplaced:
        raise ConfigError(f"could not replace {', '.join(unreplaced)} in the deck's text")
    if writer.external:
        writer.notes.append(
            f"files outside the deck's folder are copied to {EXTERNAL}/ "
            f"({', '.join(sorted(writer.external))})"
        )
    return Template(HEADER + body, writer.files, writer.notes)


class _Writer:
    def __init__(
        self,
        deck: Path,
        chosen: Mapping[str, StateProperty],
        inline: set[Path],
        aliases: Mapping[str, str],
    ) -> None:
        self.deck = deck
        self.aliases = aliases
        self.chosen = chosen
        self.inline = inline
        self.files: dict[str, Path] = {}
        self.notes: list[str] = []
        self.replaced: set[str] = set()
        self.external: set[str] = set()
        self.section = "RUNSPEC"

    def process(self, path: Path) -> str:
        out: list[str] = []
        lines = path.read_text(errors="replace").splitlines()
        index = 0
        while index < len(lines):
            line = lines[index]
            code = line.split("--", 1)[0].strip()
            keyword = _KEYWORD.match(code)
            name = keyword.group(1) if keyword else None
            if name in SECTIONS:
                self.section = name
            if name == "PATHS":
                # Its aliases are resolved here: every INCLUDE becomes ${case_dir}/<path>.
                while index + 1 < len(lines) and lines[index + 1].split("--", 1)[0].strip() != "/":
                    index += 1
                out.append("-- (PATHS: its aliases resolved by APEX's converter)")
                index += 2
                continue
            if name == "INCLUDE":
                index, record = _record(lines, index + 1)
                out.append(self._include(record, path))
                continue
            if (
                name is not None
                and name in self.chosen
                and name not in self.replaced
                and self.section in PROPERTY_SECTIONS
            ):
                index, _ = _record(lines, index + 1)
                self.replaced.add(name)
                out.append(_loop(name, self.chosen[name]))
                continue
            out.append(_escaped(line))
            index += 1
        return "\n".join(out) + "\n"

    def _include(self, record: str, including: Path) -> str:
        target = resolve_include(record, including, self.deck, self.aliases)
        if not isinstance(target, Path) or not target.is_file():
            self.notes.append(f"INCLUDE {record.strip()[:80]} was not found: kept as it is")
            return _escaped(f"INCLUDE\n {record} /")
        target = target.resolve()
        if target in self.inline:
            size = target.stat().st_size
            if size > INLINE_NOTE_BYTES:
                self.notes.append(f"{target.name} ({size // 1_000_000} MB) is inlined into the "
                                  "template: rendering each member is slower")  # fmt: skip
            return f"-- (inlined from {target.name} by APEX's converter)\n{self.process(target)}"
        relative = self._relative(target)
        known = self.files.get(relative)
        if known is not None and known != target:
            raise ConfigError(f"two included files would be copied to {relative}")
        self.files[relative] = target
        return f"INCLUDE\n '${{case_dir}}/{relative}' /"

    def _relative(self, target: Path) -> str:
        try:
            return target.relative_to(self.deck.parent).as_posix()
        except ValueError:
            self.external.add(target.name)
            return f"{EXTERNAL}/{target.name}"


def _record(lines: list[str], index: int) -> tuple[int, str]:
    """Return the index after a ``/``-terminated record starting at ``lines[index]``, and it."""
    parts: list[str] = []
    quote = ""
    while index < len(lines):
        code = lines[index].split("--", 1)[0]
        end, quote = _terminator(code, quote)
        index += 1
        if end is None:
            parts.append(code.strip())
            continue
        parts.append(code[:end].strip())
        if code[end + 1 :].strip():
            raise ConfigError(f"text after a record's '/' is not supported: {code.strip()[:60]}")
        return index, " ".join(p for p in parts if p)
    raise ConfigError("a record is not terminated by '/'")


def _loop(keyword: str, state: StateProperty) -> str:
    """Return the Mako block that writes ``keyword`` from the member's state."""
    value = "math.exp(value)" if state.transform == "log" else "value"
    head = (
        f"-- {keyword}: the member's state variable {state.variable}"
        f"{' (its logarithm)' if state.transform == 'log' else ''}, written by APEX\n"
    )
    if state.kept is None:
        return (f"{head}{keyword}\n% for value in {state.variable}:\n"
                f'${{"%.10g" % {value}}}\n% endfor\n/')  # fmt: skip
    kept = f"_{state.variable}_kept"
    runs = ", ".join(f"({flag}, {count})" for flag, count in state.kept)
    return (
        f"{head}-- (cells that are 0 in the deck stay 0)\n"
        f"<%!\n{kept} = _kept([{runs}])\n%>\n"
        f"{keyword}\n% for value, kept in zip({state.variable}, {kept}):\n"
        f'${{"%.10g" % {value} if kept else "0"}}\n% endfor\n/'
    )


def runs(flags: Sequence[bool]) -> tuple[tuple[int, int], ...]:
    """Return ``flags`` as run lengths ``(flag, count)``, for :attr:`StateProperty.kept`."""
    found: list[tuple[int, int]] = []
    for flag in flags:
        value = int(bool(flag))
        if found and found[-1][0] == value:
            found[-1] = (value, found[-1][1] + 1)
        else:
            found.append((value, 1))
    return tuple(found)


def _escaped(line: str) -> str:
    """Return a deck line Mako renders as it is."""
    stripped = line.lstrip()
    if stripped.startswith(("##", "%", "<%")):
        line = "${''}" + line
    if line.rstrip().endswith("\\"):
        line += "${''}"  # a trailing backslash would join the next line in Mako
    return line
