"""Which APEX modules a contributed module reaches as it is imported.

A contributed agent's entry module is loaded while ``services.settings`` is imported, so
what it imports, directly or through packages built on APEX, may be only ``apex.core`` and
``apex.domain`` (:data:`apex.core.agents.ALLOWED_IMPORTS`). Two checks, together
independent of what a process imported first:

- :func:`module_level_imports` reads the sources: the module-level ``import`` statements of
  a module and of every module it imports from a package built on APEX (its own package,
  or another that depends on ``apex``), so a module loaded before (whose body never runs
  again) is judged as one loaded now. Not read: function bodies (they run when called),
  ``if TYPE_CHECKING:`` and ``if __name__ == "__main__":`` blocks (they never run on
  import).
- :func:`refusing_imports` sees every ``import`` statement the loading thread runs (the
  entry point's function too) and refuses a forbidden one before it runs, so a misbehaving
  contributor neither half imports the services nor waits on their import lock.

Both fail open where they cannot see: a module without a ``.py`` source (an extension, a
zip, an import-hook editable install) is taken as importing nothing, and
``importlib.import_module`` in a member's code is not seen. The member's own import-linter
contract is the static rule; these keep a contributor that breaks it from half loading.
"""

import ast
import builtins
import functools
import importlib.machinery
import importlib.metadata
import importlib.util
import re
import sys
import threading
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from pathlib import Path
from types import ModuleType

#: An import of ``imported`` by ``importer`` (module names).
type Edge = tuple[str, str]


def forbidden(name: str, allowed: Sequence[str]) -> bool:
    """Whether ``name`` is an APEX module outside ``allowed`` (``apex`` itself is not)."""
    return name.startswith("apex.") and not any(
        name == prefix or name.startswith(f"{prefix}.") for prefix in allowed
    )


def module_level_imports(modules: Iterable[str], allowed: Sequence[str]) -> list[Edge]:
    """Return the forbidden imports ``modules`` make as they are imported, read from sources.

    Follows the modules of their own packages and of the packages built on APEX
    (:func:`built_on_apex`) that they import, and the packages above each (whose
    ``__init__`` runs first), each read once, without importing any. Give only modules of
    such packages: any other's own package would be followed whole.
    """
    followed = built_on_apex() | {m.partition(".")[0] for m in modules} - {"apex"}
    pending = [n for m in modules if m.partition(".")[0] != "apex" for n in (*_parents(m), m)]
    done: set[str] = set()
    edges: set[Edge] = set()
    while pending:
        name = pending.pop()
        if name in done:
            continue
        done.add(name)
        source = _source(name)
        if source is None:
            continue
        for target in _imports_of(name, source, allowed):
            if forbidden(target, allowed):
                edges.add((name, target))
            elif target.partition(".")[0] in followed:
                pending += [*_parents(target), target]
    return sorted(edges)


def refusing_imports[T](load: Callable[[], T], allowed: Sequence[str]) -> tuple[T, set[str]]:
    """Return ``load()`` and the modules its thread imported, refusing forbidden ones.

    Raises
    ------
    ImportError
        Before a forbidden module is imported, naming the importer.
    """
    original = builtins.__import__
    thread = threading.get_ident()
    seen: set[str] = set()
    # Off once the load is over, even if the hook stays installed (someone who patched
    # __import__ meanwhile may restore ours): it then only passes through.
    active = [True]
    checking = [False]

    def refusing(
        name: str,
        globals: Mapping[str, object] | None = None,
        locals: Mapping[str, object] | None = None,
        fromlist: Sequence[str] | None = (),
        level: int = 0,
    ) -> ModuleType:
        # Not while checking: the check itself must never re-enter the hook.
        if active[0] and not checking[0] and threading.get_ident() == thread:
            checking[0] = True
            try:
                importer = str((globals or {}).get("__name__", "?"))
                targets = list(_targets(name, globals, fromlist, level, allowed))
            finally:
                checking[0] = False
            for target in targets:
                if forbidden(target, allowed):
                    raise ImportError(f"{importer} imports {target}")
                seen.add(target)
        return original(name, globals, locals, fromlist, level)

    builtins.__import__ = refusing
    try:
        return load(), seen
    finally:
        active[0] = False
        # Only ours: a patch made meanwhile by someone else stays.
        if builtins.__import__ is refusing:
            builtins.__import__ = original


def _targets(
    name: str,
    globals: Mapping[str, object] | None,
    fromlist: Sequence[str] | None,
    level: int,
    allowed: Sequence[str],
) -> Iterator[str]:
    if level > 0:
        package = str((globals or {}).get("__package__") or "")
        try:
            name = importlib.util.resolve_name("." * level + name, package)
        except ImportError:
            return
    yield name
    if not forbidden(name, allowed):  # its names add nothing to a refusal
        yield from _submodules(name, fromlist or ())


def _submodules(base: str, names: Iterable[str]) -> Iterator[str]:
    # `from apex import services` imports a module, `from apex import __version__` a name.
    for item in names:
        if item != "*" and _is_module(f"{base}.{item}"):
            yield f"{base}.{item}"


def _is_module(name: str) -> bool:
    return name in sys.modules or _spec(name) is not None


def _spec(name: str) -> importlib.machinery.ModuleSpec | None:
    """Return module ``name``'s spec from the path, never importing anything.

    ``importlib.util.find_spec`` would import the packages above ``name``, running their
    ``__init__`` unchecked; a module the path finders cannot place (an import hook) is
    taken as none.
    """
    head, *rest = name.split(".")
    loaded = sys.modules.get(head)
    try:
        spec = (
            loaded.__spec__ if loaded is not None and loaded.__spec__ is not None
            else importlib.machinery.PathFinder.find_spec(head)
        )  # fmt: skip
        for depth in range(1, len(rest) + 1):
            # A loaded package's __path__ (which pkgutil.extend_path may have widened),
            # else its spec's locations.
            parent = sys.modules.get(".".join([head, *rest[: depth - 1]]))
            locations = getattr(parent, "__path__", None) or (
                None if spec is None else spec.submodule_search_locations
            )
            if not locations:
                return None
            fullname = ".".join([head, *rest[:depth]])
            module = sys.modules.get(fullname)
            spec = (
                module.__spec__ if module is not None and module.__spec__ is not None
                else importlib.machinery.PathFinder.find_spec(fullname, list(locations))
            )  # fmt: skip
    except (ImportError, ValueError, OSError):
        return None
    return spec


@functools.cache
def built_on_apex() -> frozenset[str]:
    """Return the top-level modules of installed distributions that depend on ``apex``."""
    dependents = set()
    for dist in importlib.metadata.distributions():
        requires = dist.requires or []
        if any(re.match(r"apex(?![\w.-])", r.strip(), re.IGNORECASE) for r in requires):
            dependents.add(dist.metadata["Name"])
    tops: set[str] = set()
    for dist in importlib.metadata.distributions():
        if dist.metadata["Name"] not in dependents:
            continue
        # Their own files only (every distribution's RECORD would cost a second or so).
        for file in dist.files or ():
            head = file.parts[0] if file.parts else ""
            if head.endswith((".dist-info", ".data", ".pth")) or head in ("..", "__pycache__"):
                continue
            tops.add(head.removesuffix(".py"))
        # An editable install's files are a .pth only: its packages are those its entry
        # points name (every workspace member names its own in `apex.audience`).
        tops.update(e.module.partition(".")[0] for e in dist.entry_points)
    return frozenset(tops)


def _parents(name: str) -> list[str]:
    parts = name.split(".")
    return [".".join(parts[:depth]) for depth in range(1, len(parts))]


def _source(name: str) -> Path | None:
    """Return the ``.py`` file of module ``name``, without importing anything."""
    loaded = getattr(sys.modules.get(name), "__file__", None)
    origin = loaded if loaded is not None else getattr(_spec(name), "origin", None)
    return Path(origin) if isinstance(origin, str) and origin.endswith(".py") else None


def _imports_of(name: str, source: Path, allowed: Sequence[str]) -> list[str]:
    """Return the modules ``source``'s statements run on import import."""
    try:
        tree = ast.parse(source.read_text(), str(source))
    except (OSError, SyntaxError, ValueError, RecursionError, MemoryError):
        return []
    package = name if source.name == "__init__.py" else name.rpartition(".")[0]
    found: list[str] = []
    for node in _run_on_import(tree.body):
        if isinstance(node, ast.Import):
            found += [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ""
            if node.level:
                try:
                    base = importlib.util.resolve_name("." * node.level + base, package)
                except ImportError:
                    continue
            found.append(base)
            if not forbidden(base, allowed):
                found += _submodules(base, (alias.name for alias in node.names))
    return found


def _run_on_import(body: list[ast.stmt]) -> Iterator[ast.stmt]:
    """Return the statements of ``body`` that run on import, nested ones included.

    Statements only, with an explicit stack: an expression is never walked, so no nesting
    depth can exhaust the recursion limit.
    """
    stack = list(reversed(body))
    while stack:
        node = stack.pop()
        yield node
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            continue
        blocks: list[list[ast.stmt]] = []
        if isinstance(node, ast.If):
            blocks = [node.orelse] if _never_on_import(node.test) else [node.body, node.orelse]
        elif isinstance(node, ast.Try | ast.TryStar):
            blocks = [node.body, *(h.body for h in node.handlers), node.orelse, node.finalbody]
        elif isinstance(node, ast.For | ast.AsyncFor | ast.While):
            blocks = [node.body, node.orelse]
        elif isinstance(node, ast.With | ast.AsyncWith | ast.ClassDef):
            blocks = [node.body]
        elif isinstance(node, ast.Match):
            blocks = [case.body for case in node.cases]
        for block in reversed(blocks):
            stack += reversed(block)


def _never_on_import(test: ast.expr) -> bool:
    # `if TYPE_CHECKING:` / `if typing.TYPE_CHECKING:`, and `if __name__ == "__main__":`.
    if isinstance(test, ast.Name) and test.id == "TYPE_CHECKING":
        return True
    if isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING":
        return True
    return (
        isinstance(test, ast.Compare)
        and isinstance(test.left, ast.Name)
        and test.left.id == "__name__"
        and len(test.comparators) == 1
        and isinstance(test.comparators[0], ast.Constant)
        and test.comparators[0].value == "__main__"
    )
