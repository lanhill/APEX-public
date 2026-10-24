"""Run identity: what a run computes, as a hash.

A run's id is the SHA-256 of:

- its resolved configuration, without the fields that only name or place it (``[run]
  name`` and ``output_dir``), in which every path to an existing file is replaced by the
  SHA-256 of that file's content, and every directory by the hashes of its files;
- the **model directory** of a deck template (the template's folder, where its INCLUDEs
  live): every regular file in it, except run outputs (``runs/``, ``predictions/``,
  ``report/`` and the like), so that editing an included grid, PVT or schedule file
  changes the id;
- an executor's **locations** (its ``location_options``, e.g. the ssh executor's
  ``local_case_dir``): folders that only say where things are, kept as the path itself,
  never hashed; the inputs in them count through the options that name them;
  an executor not installed here has every option kept as given, never hashed;
- the **code**: the APEX version and commit (with the diff of an uncommitted working
  tree), and the simulator's version where it reports one (``flow --version``), so an
  upgrade or a bug fix never reuses results of the old code.

Two runs with the same id compute the same thing, whatever they are called and wherever
their inputs live. Options that do not change the numbers (e.g. ``[execution]
workers``) still count: at worst an identical run is not reused (conservative).
"""

import functools
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import apex
from apex.domain.config import HistoryMatchConfig
from apex.domain.optimization import OptimizationConfig
from apex.storage.provenance import file_hash, git

#: Length of the id's hex prefix used in names (the full hash is the id).
SHORT = 12

#: Folders under a model directory that hold outputs, not inputs.
OUTPUT_DIRS = frozenset(
    {"runs", "predictions", "report", "comparison", "data_split", "results", "__pycache__"}
)


def _listing(directory: Path) -> tuple[tuple[str, int, int], ...]:
    files = []
    for path in sorted(directory.rglob("*")):
        parts = path.relative_to(directory).parts
        if any(p in OUTPUT_DIRS or p.startswith(".") for p in parts) or not path.is_file():
            continue
        stat = path.stat()
        files.append((str(path.relative_to(directory)), stat.st_size, stat.st_mtime_ns))
    return tuple(files)


@functools.lru_cache(maxsize=64)
def _tree_hash(directory: str, listing: tuple[tuple[str, int, int], ...]) -> str:
    # Cached by the listing (names, sizes, modification times): a study validating many
    # cells of one case hashes its model directory once.
    digest = hashlib.sha256()
    for name, _, _ in listing:
        digest.update(name.encode())
        digest.update(file_hash(Path(directory) / name).encode())
    return digest.hexdigest()


def tree_hash(directory: Path) -> str:
    """SHA-256 of every input file under ``directory`` (outputs and hidden files excluded)."""
    return _tree_hash(str(directory), _listing(directory))


def _executor_locations(document: dict[str, Any]) -> dict[str, Any]:
    """Take the executor's location options out of ``document``, and return them as given.

    A location (the ssh executor's ``local_case_dir``, the whole case-studies folder) is not
    an input: hashing its tree would read every file under it at each validation, and change
    the id with any unrelated file there (the coder's review).
    """
    # Composition registers every executor: a caller that has not imported it must not
    # fall back to hashing a location (imported here: composition itself uses this module).
    import apex.services.composition  # noqa: F401
    from apex.core.errors import UnknownPluginError
    from apex.core.plugins import executors

    execution = document.get("execution")
    if not isinstance(execution, dict) or not isinstance(execution.get("options"), dict):
        return {}
    options = execution["options"]
    try:
        names = getattr(executors.factory(str(execution.get("name"))), "location_options", ())
    except UnknownPluginError:
        # Not installed here (its package missing): which options are locations
        # is unknown, so none is read from disk; every value counts as given (a whole
        # case-studies folder would otherwise be hashed, the coder's review of phase 1).
        # The id then differs from an installed environment's (raw strings, not resolved
        # paths), but a run with an unknown executor cannot execute: no wrong reuse follows.
        return {name: options.pop(name) for name in sorted(options)}
    # Resolved, so that two spellings of one folder give one id.
    return {
        name: None if value is None else str(Path(str(value)).resolve())
        for name in sorted(names)
        if name in options
        for value in [options.pop(name)]
    }


def _content(value: Any) -> Any:  # noqa: ANN401 (JSON values)
    if isinstance(value, dict):
        return {k: _content(v) for k, v in sorted(value.items())}
    if isinstance(value, list):
        return [_content(v) for v in value]
    if isinstance(value, str) and ("/" in value or value.endswith((".npz", ".npy", ".csv"))):
        path = Path(value)
        if path.is_file():
            return {"file_sha256": file_hash(path)}
        if path.is_dir():
            return {"dir_sha256": tree_hash(path)}
    return value


@functools.lru_cache(maxsize=1)
def code_version() -> dict[str, str | None]:
    """Return the APEX code: version, commit and, for an uncommitted tree, its diff hash."""
    diff = git("diff", "HEAD")
    return {
        "apex": apex.__version__,
        "commit": git("rev-parse", "HEAD"),
        "uncommitted": hashlib.sha256(diff.encode()).hexdigest() if diff else None,
    }


@functools.lru_cache(maxsize=16)
def _simulator_version(name: str, options_json: str) -> str | None:
    from apex.core.plugins import simulators

    try:
        simulator = simulators.create(name, **json.loads(options_json))
    except (TypeError, ValueError):
        return None
    version = getattr(simulator, "version", None)
    return None if version is None else version()


def _engine_version(name: str) -> str | None:
    from importlib import metadata

    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def run_id(config: HistoryMatchConfig, engine: Mapping[str, Any] | None = None) -> str:
    """Return the id of the run ``config`` describes (paths must be resolved).

    ``engine`` is the method of another engine than APEX's (``EngineMethod``
    as JSON): it and that engine's version count, so an ERT cell never shares an APEX
    cell's run. APEX's own runs keep the ids they always had.
    """
    # An executor's defaults count whichever route the configuration came by.
    from apex.services.composition import executor_defaults

    config = executor_defaults(config)
    document = json.loads(config.model_dump_json())
    document["run"].pop("name", None)
    document["run"].pop("output_dir", None)
    locations = _executor_locations(document)
    template = config.simulator.options.get("deck_template_file")
    identity = {
        "config": _content(document),
        "model_dir": None if template is None else tree_hash(Path(str(template)).parent),
        "code": code_version(),
        "simulator_version": _simulator_version(
            config.simulator.name, json.dumps(config.simulator.options, sort_keys=True)
        ),
    }
    if locations:
        identity["locations"] = locations
    if engine is not None:
        identity["engine"] = {**engine, "version": _engine_version(str(engine["engine"]))}
    canonical = json.dumps(identity, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def optimization_id(config: OptimizationConfig) -> str:
    """Return the id of the optimization ``config`` describes (paths must be resolved).

    As :func:`run_id`: the configuration without its name and place, the folder of its deck
    template (if any), and the code's version.
    """
    document = json.loads(config.model_dump_json())
    document["optimization"].pop("name", None)
    document["optimization"].pop("output_dir", None)
    locations = _executor_locations(document)
    template = (
        None if config.simulator is None else config.simulator.options.get("deck_template_file")
    )
    identity = {
        "config": _content(document),
        "model_dir": None if template is None else tree_hash(Path(str(template)).parent),
        "code": code_version(),
    }
    if locations:
        identity["locations"] = locations
    canonical = json.dumps(identity, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def owned_run_id(content: str, submitted_by: str | None) -> str:
    """Return a run's catalogue id: its content id, made a guest's own.

    The owner's runs keep their content id (:func:`run_id`). A guest's run gets an id of its
    own, so a guest never reuses, nor sees, another person's run of an identical
    configuration; it runs again in their folder, and their own resubmissions reuse it.
    """
    if not submitted_by or not submitted_by.startswith("guest:"):
        return content
    return hashlib.sha256(f"{content}|{submitted_by}".encode()).hexdigest()
