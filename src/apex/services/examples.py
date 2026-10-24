"""The catalogue of examples and toy problems, grouped by workflow (``domain/examples.py``).

The examples root (``[examples] root``; by default the repository's ``examples/`` when APEX
runs from a checkout) holds folders with an ``example.toml`` each. :func:`find_examples`
lists them; :func:`copy_example` copies one into a user's folder so that it can be edited and
run like any configuration of theirs:

- every file is copied except deck templates (``.mako``): Python, which a copy must not let
  a user edit into a template; a copied configuration names them in the examples
  root instead, which the file checks trust as the server's (``access.PathRules``);
- programs and notebooks (``.py``, ``.ipynb``) are not copied either: a copy is for running
  and editing configurations, and these would only take a user's space;
- references to files outside the example's folder are made absolute the same way, in every
  copied ``.toml``;
- a copy never overwrites: a second copy of the same example gets ``-2``, ``-3``, ... (the
  folder is created atomically, so two copies at once never merge), and so do the names of
  its configurations' runs, optimizations, studies and predictions (``linear_esmda-2``), so
  that a copy's run never collides with an earlier copy's in the user's ``runs/`` folder
  (past any run name already taken there), with plain ``"runs/<name>"`` references to them;
- a deck template reads its include files beside it, in the examples root: a copy's data
  files are the user's own record, not what its deck reads.
"""

import datetime as dt
import json
import logging
import os
import re
import shutil
import tomllib
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import ValidationError

import apex
from apex.core.errors import ConfigError, validation_problems
from apex.domain.examples import ExampleManifest
from apex.services.settings import ExamplesSection

logger = logging.getLogger(__name__)

MANIFEST = "example.toml"
#: Written in each copy's folder: which example it is and when it was copied, so that guest
#: retention finds and dates copies as it does submissions.
COPY_MARKER = ".apex-example.json"
#: Templates: code, never copied; they stay in the examples root, which is trusted for them.
CODE_SUFFIXES = (".mako",)
#: Other files never copied (programs, notebooks).
SKIPPED_SUFFIXES = (".py", ".ipynb")
#: Folders never copied (results of earlier runs, caches).
SKIPPED_FOLDERS = frozenset({"runs", "results", "__pycache__", ".ipynb_checkpoints", "report"})


@dataclass(frozen=True)
class Example:
    """One example: its id (its folder relative to the root), folder and manifest."""

    id: str
    folder: Path
    manifest: ExampleManifest


@dataclass(frozen=True)
class CopiedExample:
    """Where an example was copied: its folder, and each entry's configuration there."""

    folder: Path
    configs: list[tuple[str, str | None, Path]]  # (workflow, title, path)


def examples_root(settings: ExamplesSection) -> Path | None:
    """Return the examples root: configured, else the checkout's ``examples/``, else none."""
    if not settings.enabled:
        return None
    if settings.root:
        return Path(settings.root).expanduser().resolve()
    checkout = Path(apex.__file__).resolve().parents[2]  # src/apex/__init__.py -> the repo
    folder = checkout / "examples"
    if (checkout / "pyproject.toml").exists() and folder.is_dir():
        return folder
    return None


def read_manifest(path: Path) -> ExampleManifest:
    """Read and validate an ``example.toml``.

    Raises
    ------
    ConfigError
        If it cannot be read, does not match the schema, or names a configuration that is
        missing or outside its folder.
    """
    try:
        manifest = ExampleManifest.model_validate(tomllib.loads(path.read_text("utf-8")))
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise ConfigError(f"cannot read {path}: {error}") from error
    except ValidationError as error:
        raise ConfigError(f"invalid {path}:\n{validation_problems(error)}") from error
    folder = path.parent.resolve()
    for entry in manifest.example.entries:
        config = (folder / entry.config).resolve()
        if folder not in config.parents or not config.is_file() or config.suffix != ".toml":
            raise ConfigError(f"{path}: entry {entry.config!r} is not a .toml file of its folder")
    return manifest


def find_examples(root: Path, *, strict: bool = False) -> list[Example]:
    """Return every example under ``root``, by id.

    An invalid manifest is logged and left out, or raised with ``strict`` (the tests).
    """
    found: list[Example] = []
    for path in sorted(root.rglob(MANIFEST)):
        if any(part in SKIPPED_FOLDERS for part in path.relative_to(root).parts):
            continue
        try:
            manifest = read_manifest(path)
        except ConfigError as error:
            if strict:
                raise
            logger.warning("example skipped: %s", error, extra={"event": "example_invalid"})
            continue
        found.append(Example(path.parent.relative_to(root).as_posix(), path.parent, manifest))
    return found


def get_example(root: Path, example_id: str) -> Example:
    """Return one example by id.

    The id is a plain relative path (no ``..``, ``.``, empty or absolute parts), and the
    example's real folder lies in the root: the returned id is the canonical one, so a copy
    is placed by it and nothing else.

    Raises
    ------
    KeyError
        If there is none (or the id leaves the root, or names it by a detour).
    """
    parts = example_id.split("/")
    if Path(example_id).is_absolute() or any(p in ("", ".", "..") or "\\" in p for p in parts):
        raise KeyError(f"no example {example_id!r}")
    base = root.resolve()
    folder = (base / example_id).resolve()
    if base not in folder.parents or not (folder / MANIFEST).is_file():
        raise KeyError(f"no example {example_id!r}")
    canonical = folder.relative_to(base).as_posix()
    if canonical != example_id:  # a link inside the root: listed under its own place only
        raise KeyError(f"no example {example_id!r}")
    return Example(canonical, folder, read_manifest(folder / MANIFEST))


def example_size(example: Example) -> int:
    """Return the bytes a copy of ``example`` takes."""
    return sum(path.stat().st_size for path in example_files(example.folder))


def copy_example(
    example: Example, destination: Path, runs_dir: Path | None = None
) -> CopiedExample:
    """Copy an example into ``destination/<id>`` (``-2``, ``-3``, ... if taken); see above.

    ``runs_dir`` is the folder its runs would go to (the web UI's ``UI/<user>/runs``): a run
    name that a run there already has is suffixed too, past every taken one.

    Raises
    ------
    ValueError
        If the id would place the copy outside ``destination``.
    """
    base = destination.resolve()
    first = (base / example.id).resolve()
    if base not in first.parents:
        raise ValueError(f"example id {example.id!r} leaves the destination")
    first.parent.mkdir(parents=True, exist_ok=True)
    target, n = first, 2
    while True:
        try:
            target.mkdir()  # atomic: a concurrent copy takes the next name
            break
        except FileExistsError:
            if target == first and _holds_only_nested_copies(target):
                # Made as the parent of a nested example's copy (optimization/3well before
                # optimization): this example's own name, not "-2" (the release tester).
                break
            target = first.with_name(f"{first.name}-{n}")
            n += 1
    for source in example_files(example.folder):
        copied = target / source.relative_to(example.folder)
        copied.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, copied)
    suffix = _names_suffix(target, first, runs_dir)
    renamed = _run_names(target) if suffix else set()
    for path in sorted(target.rglob("*.toml")):
        config = path.relative_to(target).as_posix()
        try:
            text = _absolute_references(path.read_text("utf-8"), example, config)
        except tomllib.TOMLDecodeError:
            continue  # not a configuration; copied as it is
        text = _own_case_dir(text, example, target, path)
        if suffix and path.name != MANIFEST:
            text = _suffixed_names(text, suffix, example, config)
            text = _renamed_references(text, renamed, suffix, example, config)
        path.write_text(text, "utf-8")
    configs = [(entry.workflow, entry.title, target / entry.config)
               for entry in example.manifest.example.entries]  # fmt: skip
    created = dt.datetime.now(dt.UTC).isoformat(timespec="seconds")
    (target / COPY_MARKER).write_text(json.dumps({"id": example.id, "created": created}) + "\n")
    return CopiedExample(target, configs)


def _run_names(folder: Path) -> set[str]:
    """Return the ``[run]`` names of the configurations in ``folder`` (unparsable ones: none)."""
    names = set()
    for path in folder.rglob("*.toml"):
        try:
            table = tomllib.loads(path.read_text("utf-8")).get("run")
        except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError):
            continue
        name = table.get("name") if isinstance(table, dict) else None
        if isinstance(name, str):
            names.add(name)
    return names


def _names_suffix(target: Path, first: Path, runs_dir: Path | None) -> str:
    """Return the suffix a copy's names get: ``""`` when nothing they place would collide.

    The first copy keeps the example's names (as its documentation gives them), unless a
    run of one of them is already in ``runs_dir``. A later copy (folder ``<id>-n``) takes
    ``-n``, or the next number no run in ``runs_dir`` has for any of its run names.
    """
    names = _run_names(target)
    # Names other copies of the example already give their runs, run or not yet run (two
    # copies made before either ran would otherwise share names).
    siblings = {
        name
        for other in target.parent.glob(f"{first.name}*")
        if other != target
        and other.is_dir()
        and (other.name == first.name or other.name.startswith(f"{first.name}-"))
        for name in _run_names(other)
    }

    def taken(name: str) -> bool:
        return name in siblings or (runs_dir is not None and (runs_dir / name).exists())

    suffixed = target != first
    if not suffixed and not any(taken(n) for n in names):
        return ""
    n = int(target.name.rsplit("-", 1)[1]) if suffixed else 2
    while any(taken(f"{name}-{n}") for name in names):
        n += 1
    return f"-{n}"


def _renamed_references(
    text: str, renamed: set[str], suffix: str, example: Example, config: str
) -> str:
    """Return a configuration whose plain references to a renamed run name it too.

    A string ``"runs/<name>"`` (or ``"./runs/<name>"``, or a path under it), such as a
    prediction's ``run_dir``, follows the run's new name. A reference written otherwise is
    left as it is, with a warning.
    """
    for name in sorted(renamed):
        quoted = re.compile(rf"""(["'])((?:\./)?runs/){re.escape(name)}(?=(?:/[^"'\n]*)?\1)""")
        # A template, not a function: "\g<1>" and "\g<2>" are the quote and the prefix.
        text = quoted.sub("\\g<1>\\g<2>" + f"{name}{suffix}".replace("\\", "\\\\"), text)
        other = re.compile(rf"runs/{re.escape(name)}(?![\w.-])")
        left = [line for line in text.splitlines()
                if other.search(line) and not line.lstrip().startswith("#")]  # fmt: skip
        if left:
            logger.warning("example %s: %s names run %r in a way not renamed: %s",
                           example.id, config, name, left[0].strip())  # fmt: skip
    return text


#: The sections whose ``name`` places what a configuration writes (a run's ``runs/<name>``).
NAMED_SECTIONS = ("run", "optimization", "study", "prediction")


def _suffixed_names(text: str, suffix: str, example: Example, config: str) -> str:
    """Return a configuration whose run (optimization, study, prediction) names end in ``suffix``.

    The text is edited in place, comments kept: the first ``name = "..."`` line of each
    section of :data:`NAMED_SECTIONS` (a table header ``[run]``), checked by parsing the
    result. A name written otherwise (an inline table) is left as it is, with a warning.
    """
    document = tomllib.loads(text)
    for section in NAMED_SECTIONS:
        table = document.get(section)
        name = table.get("name") if isinstance(table, dict) else None
        if not isinstance(name, str):
            continue
        header = re.search(rf"(?m)^[ \t]*\[[ \t]*{section}[ \t]*\][ \t]*(#.*)?$", text)
        end = len(text)
        if header is not None:
            following = re.search(r"(?m)^[ \t]*\[", text[header.end() :])
            end = header.end() + following.start() if following else len(text)
        line = None if header is None else re.compile(
            rf"""(?m)^([ \t]*name[ \t]*=[ \t]*)(["']){re.escape(name)}\2"""
        ).search(text, header.end(), end)  # fmt: skip
        if line is None:
            logger.warning("example %s: %s keeps its [%s] name %r", example.id, config,
                           section, name)  # fmt: skip
            continue
        changed = (text[: line.start()] + f"{line.group(1)}{line.group(2)}{name}{suffix}"
                   f"{line.group(2)}" + text[line.end():])  # fmt: skip
        try:
            renamed = tomllib.loads(changed)[section]["name"]
        except (tomllib.TOMLDecodeError, KeyError, TypeError):
            renamed = None
        if renamed != f"{name}{suffix}":
            logger.warning("example %s: %s keeps its [%s] name %r", example.id, config,
                           section, name)  # fmt: skip
            continue
        text = changed
    return text


def _holds_only_nested_copies(folder: Path) -> bool:
    """Whether ``folder`` holds nothing but sub-folders that are examples' copies.

    Such a folder was made as the parent of a nested example's copy; the example of its own
    name copies into it, beside them (its files leave nested examples out).
    """
    if (folder / COPY_MARKER).exists():
        return False
    entries = list(folder.iterdir())
    # Not empty: an empty one may be a concurrent copy that has just made its folder.
    return bool(entries) and all(e.is_dir() and _is_copy_tree(e) for e in entries)


def _is_copy_tree(folder: Path) -> bool:
    """Whether ``folder`` is an example's copy, or holds only such copies."""
    return (folder / COPY_MARKER).exists() or _holds_only_nested_copies(folder)


def example_files(folder: Path) -> Iterator[Path]:
    """Yield the files an example copies: data, not code nor run output, nor nested examples."""
    # A sub-folder with a manifest of its own is another example: not part of this one.
    nested = [m.parent for m in folder.rglob(MANIFEST) if m.parent != folder]
    for path in sorted(folder.rglob("*")):
        relative = path.relative_to(folder).parts
        if any(part in SKIPPED_FOLDERS for part in relative) or not path.is_file():
            continue
        if any(n in path.parents for n in nested):
            continue
        if path.suffix in CODE_SUFFIXES or path.suffix in SKIPPED_SUFFIXES:
            continue
        if path.is_symlink() and folder not in path.resolve().parents:
            continue  # a link out of the example is not part of it
        yield path


def _absolute_references(text: str, example: Example, config: str) -> str:
    """Return a configuration's text with its references to files not copied made absolute.

    A string value naming an existing file, relative to the configuration's folder, that is
    a template or lies outside the example's folder, becomes the file's absolute path in the
    examples root; the text is otherwise kept (comments included).
    """
    base = (example.folder / config).parent
    folder = example.folder.resolve()
    replacements: dict[str, str] = {}
    for value in _strings(tomllib.loads(text)):
        if Path(value).is_absolute():
            continue
        candidate = (base / value).resolve()
        if not candidate.is_file():
            continue
        outside = folder not in candidate.parents
        if outside or candidate.suffix in CODE_SUFFIXES:
            replacements[value] = str(candidate)
    for value, absolute in replacements.items():
        pattern = re.compile(rf'(["\']){re.escape(value)}\1')
        text = pattern.sub(_quoted(absolute), text)
    return text


def _own_case_dir(text: str, example: Example, target: Path, path: Path) -> str:
    """Return a copied configuration that reads its deck's includes from the copy.

    Its deck template stays in the examples root (code; ``_absolute_references``), where
    ``${case_dir}`` would name the original's folder: the copy's data files would be ignored.
    The simulator's ``case_dir`` option names the copy's matching folder instead, relative to
    the configuration. Left as it is if the simulator sets ``case_dir`` itself, the template
    lies outside the example, or the option cannot be placed beside ``deck_template_file``.
    """
    document = tomllib.loads(text)
    simulator = document.get("simulator")
    options = simulator.get("options") if isinstance(simulator, dict) else None
    if not isinstance(options, dict) or "case_dir" in options:
        return text
    template = options.get("deck_template_file")
    folder = example.folder.resolve()
    if not isinstance(template, str) or not Path(template).is_absolute():
        return text
    if folder not in Path(template).parents:
        return text
    copied = target / Path(template).parent.relative_to(folder)
    relative = Path(os.path.relpath(copied, path.parent)).as_posix()
    pattern = re.compile(
        rf"""(deck_template_file\s*=\s*(["']){re.escape(template)}\2)([ \t]*)([,}}]?)"""
    )

    def placed(match: re.Match[str]) -> str:
        if match.group(4):  # in an inline table: another key of it
            return f'{match.group(1)}, case_dir = "{relative}"{match.group(3)}{match.group(4)}'
        line_start = text.rfind("\n", 0, match.start()) + 1
        indent = re.match(r"[ \t]*", text[line_start:]).group(0)  # type: ignore[union-attr]
        return f'{match.group(1)}\n{indent}case_dir = "{relative}"{match.group(3)}'

    changed, count = pattern.subn(placed, text, count=1)
    try:
        written = tomllib.loads(changed)["simulator"]["options"].get("case_dir")
    except (tomllib.TOMLDecodeError, KeyError, TypeError, AttributeError):
        written = None
    if count != 1 or written != relative:
        logger.warning(
            "example %s: %s keeps its template's folder for its includes", example.id,
            path.name, extra={"event": "example_case_dir_skipped", "example": example.id},
        )  # fmt: skip
        return text
    return changed


def _quoted(absolute: str) -> "Callable[[re.Match[str]], str]":
    """Return a replacement keeping the quote the configuration used."""

    def replace(match: re.Match[str]) -> str:
        return f"{match[1]}{absolute}{match[1]}"

    return replace


def _strings(value: Any) -> Iterator[str]:  # noqa: ANN401 (TOML values)
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from _strings(item)
