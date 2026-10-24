"""What a submission may read, execute and write (``[policy]`` in ``service.toml``).

A service runs outside its clients' sandboxes, so a client could otherwise make it write
anywhere (a run's ``output_dir``), read anything (a ``*_file`` option, a ``config_file``),
or execute code of its choosing: a Mako deck template is Python, and a simulator's
``executable`` (with its arguments) is any program. The policy confines all three to roots
the service's owner chooses. By default it is off (no lists set); once any list is set,
all three must be, so that a partial policy cannot leave code execution open.

- ``input_roots``: every path a configuration reads (path fields such as
  ``observations.file`` and ``case.truth_file``, and plugin options declared ``input``),
  and the files a request names (``config_file``, ``study_file``, a study's cases).
- ``code_roots``: what APEX renders or executes. Deck templates (options declared
  ``code``) must lie under them. A program (declared ``executable``) must be a path under
  them, or a name in ``executables`` that resolves (on the service's ``PATH``) under them
  or a system folder. Program arguments (declared ``arguments``) are refused: they could
  make an allowed program run a script.

**Plugin options are judged by what they declare** (``core.options``), not by
their names: each string option of a plugin's factory says whether it names a file, code,
a folder written, a program, its arguments, or text. A string in an option the plugin does
not declare is refused, so a new plugin cannot open a hole by naming its option
differently. APEX's own fields (``observations.file``, ``run.output_dir``, ...) are known
by their names and types.
- ``output_roots``: every directory a submission writes (``run.output_dir``,
  ``prediction.output_dir``).

**Trust by provenance.** A run's directory holds a snapshot (``config.json``) that resumes
and predictions execute. So a prediction's ``run_dir``, a resume and a reused run must be
directories the *service* writes: under ``output_roots`` or the service root's own
``runs/`` and ``studies/``, which the owner keeps unwritable for clients. Their snapshots
are checked as well, at submission and again by the worker just before it runs a job.

Every check runs **before** anything is resolved, built, read or written: computing a run's
identity builds its simulator and runs its executable, and validation reads its files.
Paths are resolved (symlinks, ``..``) before they are matched; the worker's re-check also
catches a symlink swapped in between submission and execution.
"""

import fnmatch
import shutil
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel

from apex.core.errors import ConfigError
from apex.core.options import Role, holds_text
from apex.domain.config import HistoryMatchConfig
from apex.domain.optimization import OptimizationConfig
from apex.domain.prediction import PredictionConfig
from apex.services.settings import PolicySection

PATH_SUFFIXES = ("_file", "_dir", "_path")
#: Options whose file APEX renders as a template (Mako templates are Python).
CODE_SUFFIXES = ("_template_file",)
#: Options that name a program APEX runs.
EXECUTABLE_KEYS = ("executable",)
#: Options that pass arguments to that program (``eclrun_args``, ``flow_args``, ...).
ARGUMENT_SUFFIXES = ("_args",)
#: Fields that name a directory APEX writes.
OUTPUT_KEYS = ("output_dir",)
#: Fields that name a run directory whose snapshot APEX executes.
SOURCE_KEYS = ("run_dir",)
#: Where a program named in ``executables`` may resolve, besides the code roots.
SYSTEM_PROGRAM_ROOTS = ("/usr", "/bin", "/sbin", "/opt")

Kind = Literal["input", "code", "output", "source", "executable", "arguments", "host", "undeclared"]


class PolicyViolation(ConfigError):
    """A submission breaks the service's ``[policy]``."""


def active(rules: PolicySection) -> bool:
    """Return whether the policy restricts anything."""
    return bool(rules.input_roots or rules.code_roots or rules.output_roots)


def _plugin_sections(config: BaseModel) -> list[tuple[str, BaseModel]]:
    """Return a configuration's plugin sections: (registry, section with name and options)."""
    from apex.services.plugins import plugin_sections

    if isinstance(config, HistoryMatchConfig):
        return list(plugin_sections(config))
    if isinstance(config, PredictionConfig):
        return [("simulators", config.simulator)]
    if isinstance(config, OptimizationConfig):
        from apex.services.plugins import objective_plugins

        sections: list[tuple[str, BaseModel]] = [
            ("optimizers", config.optimizer),
            *objective_plugins(config),
            ("executors", config.execution),
        ]
        if config.simulator is not None:
            sections.append(("simulators", config.simulator))
        return sections
    return []


def _plugin_items(
    registry: str, name: str | None, options: Mapping[str, Any]
) -> Iterator[tuple[Kind, str, str]]:
    """Yield what a plugin's options name, by their declarations."""
    from apex.core.errors import PluginError
    from apex.services.plugins import REGISTRIES, option_roles, wrapped

    roles: dict[str, Role | None] = {}
    try:
        if name is not None:
            roles = option_roles(registry, name)
            split = wrapped(registry, name, options)
            if split is not None:
                # A wrapping method (CHOP) carries its inner method's options too.
                roles = {**option_roles(registry, split[1]), **roles}
        else:
            # A prediction's changes to its run's simulator: an option's role where every
            # simulator that has the option agrees on it (the merged check is exact).
            found: dict[str, list[Role | None]] = {}
            for each in REGISTRIES[registry].names():
                for key, role in option_roles(registry, each).items():
                    found.setdefault(key, []).append(role)
            roles = {k: r[0] for k, r in found.items() if all(x == r[0] for x in r)}
    except (KeyError, PluginError):
        roles = {}  # an unknown plugin: validation names it; nothing of it is trusted here
    for key, value in options.items():
        if holds_text(value):
            yield from _role_items(roles.get(key), key, value)


def _role_items(role: Role | None, key: str, value: object) -> Iterator[tuple[Kind, str, str]]:
    if role is None:
        yield "undeclared", key, str(value)[:200]
    elif role.kind == "text":
        return
    elif role.kind == "mapping":
        records = value if isinstance(value, list | tuple) else [value]
        for record in records:
            if not isinstance(record, Mapping):
                yield "undeclared", key, str(record)[:200]
                continue
            for name, item in record.items():
                if holds_text(item):
                    yield from _role_items(role.keys.get(str(name)), f"{key}.{name}", item)
    elif role.kind == "arguments":
        arguments = value if isinstance(value, list | tuple) else [value]
        if arguments:
            yield "arguments", key, " ".join(map(str, arguments))
    else:
        kinds: dict[str, Kind] = {
            "input": "input", "code": "code", "output": "output", "executable": "executable",
            "host": "host",
        }  # fmt: skip
        for text in _strings(value):
            yield kinds[role.kind], key, text


def _strings(value: object) -> Iterator[str]:
    if isinstance(value, str | Path):
        yield str(value)
    elif isinstance(value, Mapping):
        for item in value.values():
            yield from _strings(item)
    elif isinstance(value, list | tuple | set | frozenset):
        for item in value:
            yield from _strings(item)


def _items(
    value: object, key: str = "", plugins: frozenset[int] = frozenset()
) -> Iterator[tuple[Kind, str, str]]:
    """Yield (kind, key, value) for every path, program and argument list in a configuration.

    The ``options`` of the sections in ``plugins`` (by id) are left out: :func:`_plugin_items`
    judges them by their declarations.
    """
    if isinstance(value, BaseModel):
        for name in type(value).model_fields:
            if name == "options" and id(value) in plugins:
                continue
            yield from _items(getattr(value, name), name, plugins)
    elif isinstance(value, dict):
        for name, item in value.items():
            yield from _items(item, str(name), plugins)
    elif isinstance(value, list | tuple):
        if key.endswith(ARGUMENT_SUFFIXES) and value:
            yield "arguments", key, " ".join(map(str, value))
        else:
            for item in value:
                yield from _items(item, key, plugins)
    elif isinstance(value, Path | str):
        if key in OUTPUT_KEYS:
            yield "output", key, str(value)
        elif key in SOURCE_KEYS:
            yield "source", key, str(value)
        elif key.endswith(CODE_SUFFIXES):
            yield "code", key, str(value)
        elif key in EXECUTABLE_KEYS and isinstance(value, str):
            yield "executable", key, value
        elif key.endswith(PATH_SUFFIXES) or isinstance(value, Path):
            yield "input", key, str(value)


def under(path: Path, pattern: str) -> bool:
    """Return whether ``path`` (resolved) lies under the root ``pattern`` (globs per component)."""
    parts = Path(pattern).parts
    cut = next((i for i, part in enumerate(parts) if any(c in part for c in "*?[")), len(parts))
    base = Path(*parts[:cut]).resolve()
    rest = parts[cut:]
    if path != base and base not in path.parents:
        return False
    tail = path.relative_to(base).parts
    return len(tail) >= len(rest) and all(
        fnmatch.fnmatchcase(part, glob) for part, glob in zip(tail, rest, strict=False)
    )


def _allowed(value: str | Path, roots: list[str]) -> bool:
    path = Path(value)
    return path.is_absolute() and any(under(path.resolve(), root) for root in roots)


def service_roots(root: Path) -> list[str]:
    """Return the service root's folders a submission may write and read runs from."""
    from apex.storage.optimization_directory import SERVICE_FOLDER

    base = Path(root).resolve()
    return [str(base / "runs"), str(base / "studies"), str(base / SERVICE_FOLDER)]


def _program_allowed(value: str, rules: PolicySection) -> bool:
    if "/" in value:
        return _allowed(value, rules.code_roots)
    if value not in rules.executables:
        return False
    found = shutil.which(value)  # the program the service would actually run
    return found is not None and _allowed(found, [*rules.code_roots, *SYSTEM_PROGRAM_ROOTS])


def _run_simulator(run_dir: Path) -> str | None:
    """Return the simulator a run's saved configuration names (``None`` if unreadable)."""
    import json

    try:
        saved = json.loads((Path(run_dir) / "config.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    name = (saved.get("simulator") or {}).get("name") if isinstance(saved, dict) else None
    return name if isinstance(name, str) else None


def named(config: BaseModel) -> list[tuple[Kind, str, str]]:
    """Return what a configuration names, as (kind, key, value).

    Every path, program and argument list: APEX's fields by their names, plugin options by
    their declarations.

    Used by the policy, and for web-UI users with or without one (``services/access.py``).
    """
    sections = _plugin_sections(config)
    found = list(_items(config, plugins=frozenset(id(s) for _, s in sections)))
    for registry, section in sections:
        name = getattr(section, "name", None)
        if name is None and isinstance(config, PredictionConfig):
            # Changes to the run's simulator: judged by its declarations, not by what every
            # simulator would agree on (an option two simulators declare differently was
            # refused as undeclared).
            name = _run_simulator(config.prediction.run_dir)
        options = getattr(section, "options", {}) or {}
        found += [
            (kind, f"{registry}.{name or '?'}: {key}", value)
            for kind, key, value in _plugin_items(registry, name, options)
        ]
    return found


def violations(config: BaseModel, rules: PolicySection, root: Path) -> list[str]:
    """Return why a resolved configuration breaks the policy (empty when it does not)."""
    if not active(rules):
        return []
    own = service_roots(root)
    problems = []
    found = named(config)
    for kind, key, value in found:
        if kind == "undeclared":
            problems.append(
                f"{key} = {value!r}: the plugin does not declare what this option names, so "
                "[policy] refuses it"
            )
            continue
        if kind == "input" and not _allowed(
            value, [*rules.input_roots, *rules.code_roots, *rules.output_roots, *own]
        ):
            problems.append(f"{key} = {value!r} is outside [policy] input_roots")
        elif kind == "code" and not _allowed(value, rules.code_roots):
            problems.append(
                f"{key} = {value!r} is outside [policy] code_roots (templates are code)"
            )
        elif kind == "executable" and not _program_allowed(value, rules):
            problems.append(
                f"{key} = {value!r} is neither a path under [policy] code_roots nor a name in "
                "executables that resolves under them or a system folder"
            )
        elif kind == "host" and value != "local" and value not in rules.hosts:
            problems.append(
                f"{key} = {value!r}: runs go to other machines only when [policy] hosts lists them"
            )
        elif kind == "arguments":
            problems.append(
                f"{key} = {value!r}: extra program arguments are refused under [policy] "
                "(they could make an allowed program run other code)"
            )
        elif kind == "output" and not _allowed(value, [*rules.output_roots, *own]):
            problems.append(f"{key} = {value!r} is outside [policy] output_roots")
        elif kind == "source" and not _allowed(value, [*rules.output_roots, *own]):
            problems.append(
                f"{key} = {value!r}: a run used as a source must be one the service wrote "
                "(under [policy] output_roots or the service's runs/ and studies/)"
            )
    return problems


def check(config: BaseModel, rules: PolicySection, root: Path) -> None:
    """Refuse a resolved configuration that breaks the policy.

    Raises
    ------
    PolicyViolation
        Naming every violation.
    """
    problems = violations(config, rules, root)
    if problems:
        raise PolicyViolation("refused by the service's [policy]: " + "; ".join(problems))


def check_file(path: Path, rules: PolicySection, root: Path) -> None:
    """Refuse to open a file a request names (``config_file``, a study's case, ...).

    Raises
    ------
    PolicyViolation
        If the policy is on and the file lies outside the readable roots.
    """
    readable = [*rules.input_roots, *rules.code_roots, *rules.output_roots, *service_roots(root)]
    if active(rules) and not _allowed(Path(path).absolute(), readable):
        raise PolicyViolation(
            f"refused by the service's [policy]: {path} is outside [policy] input_roots"
        )


def check_source(run_dir: Path, rules: PolicySection, root: Path) -> None:
    """Refuse a run directory that the service did not write (a resume, a reuse, a summary).

    Raises
    ------
    PolicyViolation
        If the policy is on and ``run_dir`` lies outside the output roots and the service's
        own ``runs/`` and ``studies/``.
    """
    if active(rules) and not _allowed(Path(run_dir), [*rules.output_roots, *service_roots(root)]):
        raise PolicyViolation(
            f"refused by the service's [policy]: {run_dir} is not a run the service "
            "wrote (under [policy] output_roots or the service's runs/ and studies/)"
        )


def check_job(kind: str, spec: dict[str, Any], rules: PolicySection, root: Path) -> None:
    """Re-check a queued job just before a worker runs it (what it executes, where it writes).

    Between submission and execution, a path may have been replaced (a symlink) and a run's
    snapshot may have been changed; this is the check at the time of use.

    Raises
    ------
    PolicyViolation
        If the job, as it would run now, breaks the policy.
    """
    if not active(rules):
        return
    from apex.services import composition, prediction
    from apex.storage.run_directory import RunDirectory

    if kind == "run" and spec.get("engine"):
        raise PolicyViolation(
            f"the {spec['engine'].get('engine')} engine is not allowed under a service [policy]"
        )
    if kind == "run" and "config" in spec:
        check(HistoryMatchConfig.model_validate_json(spec["config"]), rules, root)
    elif kind == "resume":
        run_dir = Path(spec["run_dir"])
        check_source(run_dir, rules, root)
        check(composition.snapshot_config(RunDirectory(run_dir)), rules, root)
    elif kind == "optimization":
        check(OptimizationConfig.model_validate_json(spec["config"]), rules, root)
    elif kind == "optimization_resume":
        from apex.storage import optimization_directory

        run_dir = Path(spec["run_dir"])
        check_source(run_dir, rules, root)
        stored = OptimizationConfig.model_validate(optimization_directory.read_config(run_dir))
        check(stored, rules, root)
    elif kind == "prediction":
        config = PredictionConfig.model_validate_json(spec["config"])
        check(config, rules, root)
        run_config = composition.snapshot_config(RunDirectory(config.prediction.run_dir))
        simulator = prediction.simulator_section(config, run_config)
        check(run_config.model_copy(update={"simulator": simulator}), rules, root)
