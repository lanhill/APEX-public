"""What web-UI users may see and use: ``<root>/ui_access.toml``.

The web UI's guests sign in with a **UI token** (``apex service token --ui [--guest NAME]``,
one per guest). The server filters every request made with one:

- **Their own work only:** a guest sees the jobs, runs, studies and predictions they created
  (each submission is annotated ``submitted_by = guest:<name>`` by the server), and their own
  chats. Other people's work, and other guests', answers ``404`` as if it did not exist.
- **An allowlist** for the rest, ``<root>/ui_access.toml``: the workspace folders they may
  open and submit from, the plugins their configurations may use, the agents, and what they
  may do. What it does not name is hidden.

The owner's token (``api_token``), the CLI, the tools over MCP and the Python client are not
affected. Example::

    # <root>/ui_access.toml
    [workspace]
    folders = ["5Spots", "Brugge/configs"]    # workspace-relative folders, with their contents

    [plugins]                             # per registry; "*" for all of one
    methods = ["esmda", "gies"]
    update_strategies = ["*"]
    localizations = ["adaptive", "distance"]
    simulators = ["opm_flow", "linear"]
    executors = ["*"]
    priors = ["*"]
    metrics = ["*"]
    # converter_skills, geomodeling_skills, variogram_estimators, chop_components:
    # display only (the Plugins page), not enforced (the user's decision, 2026-10-25)

    [agents]
    names = ["monitor", "diagnosis", "report", "plotter"]

    [actions]                             # what UI users may do; all off by default
    submit = true                         # validate and submit files of visible folders
    stop = false
    resume = false
    examples = true                       # copy the catalogue's examples into their folder

    [programs]                            # programs their configurations may name
    executables = ["flow"]                # bare names only; none by default

    [chat]                                # the assistant's models (service.toml [chat.models])
    models = ["beam", "gemma"]            # tried in order, the next when one fails; none:
                                          # no model (the agents' rules answer)

A configuration a guest submits or validates must lie in a visible folder and use only
visible plugins. What it names is judged by the declarations (``policy.named``), with
or without a ``[policy]``: input, template and source files must lie, by their real path,
in their own web-UI folder, a visible folder or one of their runs, and a template must not
be text they can write (:data:`WRITTEN_SUFFIXES`, a run's files, their folder's files but
drafted decks); a program must be a bare
name in ``[programs] executables``; program arguments, plugin options that say where to
write, and options a plugin does not declare are refused (:meth:`Access.refused_names`).
"""

import fnmatch
import logging
import tomllib
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, Field, ValidationError, model_validator

from apex.core.errors import ConfigError
from apex.domain.config import HistoryMatchConfig, _Strict

if TYPE_CHECKING:
    from apex.services.workspace import Workspace

logger = logging.getLogger(__name__)

#: The file of the list, in the services root.
ACCESS_FILE = "ui_access.toml"
#: The client name every guest submission is recorded under (for quotas and the log).
UI_CLIENT = "ui"
#: The annotation the server sets on every guest submission: who submitted it. Guests see
#: only the work that carries their own (a request cannot set it).
SUBMITTED_BY = "submitted_by"
#: The guest the original UI token belongs to.
DEFAULT_GUEST = "guest"


def submitter(guest: str) -> str:
    """Return the ``submitted_by`` value of a guest's submissions."""
    return f"guest:{guest}"


Role = Literal["owner", "ui"]


class NameRules(_Strict):
    """A list of name globs."""

    names: list[str] = Field(default_factory=list)


class FolderRules(_Strict):
    """``[workspace]``: visible workspace folders (relative, as the UI shows them)."""

    folders: list[str] = Field(default_factory=list)


class ActionRules(_Strict):
    """``[actions]``: what UI users may do beyond reading.

    All off by default, except ``delete``: a guest may delete their own finished work
    inside the workspace, as the retention would later.
    """

    submit: bool = False
    stop: bool = False
    resume: bool = False
    edit: bool = False
    delete: bool = True
    examples: bool = Field(
        default=False,
        description="Copy an example of the catalogue into their own UI folder (the Examples "
        "page; browsing it is always allowed); running it still needs submit and its plugins.",
    )


class ProgramRules(_Strict):
    """``[programs]``: programs a UI user may name in a plugin's options.

    E.g. a simulator's ``executable``, by name on the service's ``PATH``. None by default: a
    plugin then runs its own default program. Program arguments (``flow_args``, ...) are
    never theirs to set.
    """

    executables: list[str] = Field(default_factory=list)


class ChatRules(_Strict):
    """``[chat]``: the models a guest's assistant and agents answer with.

    Names of ``service.toml``'s ``[chat.models]``, tried in order. Used only when the service
    defines models; without them every chat answers with the one model as before.
    """

    models: list[str] = Field(default_factory=list)


class UIAccess(_Strict):
    """The allowlist (see the module docstring)."""

    @model_validator(mode="before")
    @classmethod
    def _no_shared_runs(cls, value: object) -> object:
        # Guests see only their own jobs, runs and studies since 2026-09-30: a list
        # that still shares some says so rather than failing obscurely.
        if isinstance(value, dict) and ({"runs", "studies"} & set(value)):
            raise ValueError(
                "[runs] and [studies] were removed: guests see only "
                "the jobs, runs and studies they created; delete these sections"
            )
        return value

    workspace: FolderRules = Field(default_factory=FolderRules)
    plugins: dict[str, list[str]] = Field(default_factory=dict)
    agents: NameRules = Field(default_factory=NameRules)
    actions: ActionRules = Field(default_factory=ActionRules)
    programs: ProgramRules = Field(default_factory=ProgramRules)
    chat: ChatRules = Field(default_factory=ChatRules)


def load_access(root: Path) -> UIAccess:
    """Read ``<root>/ui_access.toml``; without it, UI users see nothing.

    Raises
    ------
    ConfigError
        If the file cannot be read or does not match the schema.
    """
    path = Path(root) / ACCESS_FILE
    if not path.exists():
        # Nothing at all, not even ``delete``, whose default is on only within a list.
        return UIAccess(actions=ActionRules(delete=False))
    try:
        return UIAccess.model_validate(tomllib.loads(path.read_text()))
    except (OSError, tomllib.TOMLDecodeError, ValidationError) as error:
        raise ConfigError(f"invalid {path}: {error}") from error


def _any(value: str, globs: Iterable[str]) -> bool:
    return any(fnmatch.fnmatchcase(value, g) for g in globs)


class Access:
    """The list applied to one guest's view of a service root.

    A guest sees only their own work: the jobs they submitted (each carries the annotation
    ``submitted_by = guest:<name>``, which the server sets), those jobs' runs, studies and
    predictions, and the runs whose catalogue entry carries the annotation (a study's
    cells).

    Parameters
    ----------
    rules
        The allowlist (folders, plugins, agents, actions).
    records
        Every job record.
    guest
        The guest's name (their token's).
    ui_folders
        The web UI's users' folders (workspace-relative ``UI``): hidden whatever
        the list says (``folders = ["*"]`` lists them too), except ``own_folder``.
    own_folder
        The guest's own web-UI folder (workspace-relative), always theirs.
    """

    def __init__(
        self,
        rules: UIAccess,
        records: Iterable[Mapping[str, Any]],
        guest: str = DEFAULT_GUEST,
        *,
        owner: str | None = None,
        ui_folders: str | None = None,
        own_folder: str | None = None,
    ) -> None:
        self.rules = rules
        self.guest = guest
        self.ui_folders = None if ui_folders is None else ui_folders.strip("/")
        self.own_folder = None if own_folder is None else own_folder.strip("/")
        self.owner = owner or submitter(guest)
        own = [r for r in records if (r.get("annotations") or {}).get(SUBMITTED_BY) == self.owner]
        self._own_jobs = {str(r["id"]) for r in own}
        self._own_runs = {str(i) for r in own for i in r.get("run_ids") or []}
        self._own_dirs = {str(Path(d).resolve()) for r in own for d in r.get("run_dirs") or []}
        self._own_studies = {n for r in own if (n := _study_name(r))}

    # ------------------------------------------------------------------ what is visible

    def run(self, run_id: str, name: str, annotations: Mapping[str, str], path: Path) -> bool:
        """Whether a catalogue run is the guest's: one they created (revision 4).

        Not a run of the owner's or another guest's, even one a job of theirs used: a
        guest's identical resubmission runs again in their folder (``owned_run_id``).
        """
        return annotations.get(SUBMITTED_BY) == self.owner

    def study(self, name: str) -> bool:
        """Whether a study is the guest's."""
        return name in self._own_studies

    def run_dir(self, path: Path, entries: Iterable[Any]) -> bool:
        """Whether a run directory (or a prediction under it) belongs to the guest's run.

        ``entries`` are the catalogue's entries (``run_id``, ``name``, ``annotations``,
        ``path``).
        """
        target = Path(path).resolve()
        for entry in entries:
            base = Path(entry.path).resolve()
            if (target == base or base in target.parents) and self.run(
                entry.run_id, entry.name, entry.annotations, base
            ):
                return True
        return False

    def job(self, record: Mapping[str, Any], entries: Iterable[Any] = ()) -> bool:
        """Whether a job is the guest's (submitted with their token, or a cell of their study)."""
        return str(record["id"]) in self._own_jobs

    def folder(self, relative: str) -> bool:
        """Whether a workspace-relative path lies in a visible folder.

        Their own web-UI folder always; another web-UI user's never.
        """
        path = relative.strip("/")
        if self.own_folder and (path == self.own_folder or path.startswith(self.own_folder + "/")):
            return True
        if self.ui_folders is not None and (
            path == self.ui_folders or path.startswith(self.ui_folders + "/")
        ):
            return False
        for folder in self.rules.workspace.folders:
            base = folder.strip("/")
            if base in ("", "*") or path == base or path.startswith(base + "/"):
                return True
            if any(c in base for c in "*?[") and _glob_prefix(path, base):
                return True
        return False

    def on_the_way(self, relative: str) -> bool:
        """Whether a folder leads to a visible one (listed so the tree can reach it)."""
        path = relative.strip("/")
        return any(
            f.strip("/").startswith(path + "/") or not path for f in self.rules.workspace.folders
        )

    def plugin(self, registry: str, name: str) -> bool:
        """Whether a plugin is visible (and usable in UI users' configurations).

        Never a private plugin, whatever the list says (``"*"`` included): it is built for
        one site (the user, 2026-10-13).
        """
        from apex.services.plugins import is_private  # services.plugins imports every plugin

        return _any(name, self.rules.plugins.get(registry, [])) and not is_private(registry, name)

    def agent(self, name: str) -> bool:
        """Whether an agent is visible (in the list and as a chat tool)."""
        return _any(name, self.rules.agents.names)

    # ------------------------------------------------------------------ configurations

    def refused_names(self, config: BaseModel, paths: "PathRules") -> list[str]:
        """Return what a configuration names that a UI user may not (empty when nothing).

        Without a ``[policy]`` the service would read, render or execute whatever a
        configuration names: any file (``*_file`` fields, options declared ``input``), any
        deck template (``code``: Mako templates are Python) and any program (a simulator's
        ``executable``, with its arguments). For UI users every one is judged here, by the
        same declarations as the policy's, whether or not a policy is set:

        - files (and a prediction's ``run_dir``) must lie, by their real path, in a visible
          folder, the user's own web-UI folder, or one of their runs (``paths``);
        - a template must also not be text that web-UI users, or APEX from what they type,
          can write (:data:`WRITTEN_SUFFIXES`, anything in a run), and in their own web-UI
          folder it must be a draft (``drafts/``: decks ``draft_optimization`` wrote);
        - a plugin option that says where to write is refused (a run's ``output_dir`` is
          not: the server places a UI user's work in their web-UI folder);
        - a program only by a name in ``[programs] executables``; arguments never;
        - an option the plugin does not declare is refused.
        """
        from apex.services import policy

        problems: list[str] = []
        for kind, key, value in policy.named(config):
            if kind == "undeclared":
                problems.append(f"{key}: the plugin does not declare what this option names")
            elif kind == "output":
                if ": " in key:  # a plugin's option, not APEX's own output_dir
                    problems.append(f"{key}: web-UI users may not choose where a plugin writes")
            elif kind in ("input", "code", "source"):
                path = Path(value)
                where = paths.where(path) if path.is_absolute() else None
                if where is None:
                    problems.append(
                        f"{key} = {paths.shown(value)!r} is not in a folder you may use: "
                        "use a file next to the case, in a listed folder or your UI folder"
                    )
                elif kind == "code" and (
                    path.suffix.lower() in WRITTEN_SUFFIXES or where in ("run", "own")
                ):
                    problems.append(
                        f"{key} = {paths.shown(value)!r}: a template must be a deck file of "
                        "the case's folder (not a .toml, .md or .json file, nor a run's)"
                    )
            elif kind == "executable":
                if "/" in value or value not in self.rules.programs.executables:
                    problems.append(f"{key} = {value!r}: web-UI users may not choose this program")
            elif kind == "arguments":
                problems.append(f"{key}: web-UI users may not set a program's arguments")
            elif kind == "host" and value != "local":
                problems.append(f"{key} = {value!r}: web-UI users may not run on other machines")
            elif kind == "remote_program":
                problems.append(
                    f"{key} = {value!r}: web-UI users may not name programs on other machines"
                )
        return problems

    def hidden_plugins(self, config: HistoryMatchConfig) -> list[str]:
        """Return the plugins a configuration uses that UI users may not (``registry:name``)."""
        used: list[tuple[str, str]] = [
            ("methods", config.method.name),
            ("simulators", config.simulator.name),
            ("executors", config.execution.name),
        ]
        if config.method.strategy is not None:
            used.append(("update_strategies", config.method.strategy.name))
        if config.method.localization is not None:
            used.append(("localizations", config.method.localization.name))
        if config.method.damping is not None:
            used.append(("damping_schemes", config.method.damping.name))
        used.append(("priors", config.prior.generator.name))
        # Per-variable generators and joint ones: each section names its plugin.
        used += [("priors", s.name) for s in config.prior.variables.values()]
        used += [("priors", s.name) for s in config.prior.groups.values()]
        used += [("data_transforms", t.name) for t in config.data_transforms]
        used += [("parameterizations", p.name) for p in config.parameterizations]
        if config.model_error is not None:
            used.append(("model_error_models", config.model_error.name))
        from apex.services.plugins import geophysics_sections  # composition: heavy import

        used += [(r, s.name) for r, s in geophysics_sections(config)]
        used += [("metrics", m.name) for m in config.metrics or [] if m.name != "data_mismatch"]
        return sorted({f"{r}:{n}" for r, n in used if not self.plugin(r, n)})


class OwnerView(Access):
    """The owner's view in the web UI: their own work only, everything else allowed.

    In the web UI the owner and the guests do not see each other's work (the user's request,
    2026-10-01): the owner's jobs are those without a
    ``submitted_by`` annotation, and their runs, studies and predictions. With ``folder``
    (the owner's web-UI folder, ``UI/owner``; revision 3) only the work in it: a submission
    whose runs lie elsewhere (the CLI's, older runs) is not shown, and neither is a run
    outside it. A submission from the web UI that reused an identical run lying elsewhere is
    the owner's by the folder it asked to run in (``requested_dir``), but that run is still
    not listed. Folders, plugins, agents and actions are not restricted. Other clients with
    the owner's token (the CLI, the Python client, MCP) still see everything.
    """

    def __init__(self, records: Iterable[Mapping[str, Any]], folder: Path | None = None) -> None:
        super().__init__(UIAccess(actions=ActionRules(submit=True, stop=True, resume=True,
                                                      edit=True, delete=True)), [])  # fmt: skip
        self.folder_root = None if folder is None else folder.resolve()
        records = list(records)
        by_id = {str(r["id"]): r for r in records}

        def top(record: Mapping[str, Any]) -> str:
            while record.get("parent") and str(record["parent"]) in by_id:
                record = by_id[str(record["parent"])]
            return str(record["id"])

        trees: dict[str, list[Mapping[str, Any]]] = {}
        for record in records:
            trees.setdefault(top(record), []).append(record)
        own: list[Mapping[str, Any]] = []
        for head, tree in trees.items():
            if (by_id[head].get("annotations") or {}).get(SUBMITTED_BY):
                continue  # another web-UI user's submission
            dirs = [Path(d) for r in tree for d in r.get("run_dirs") or []]
            # A submission that reused a run lying elsewhere asked to run in this folder: its
            # job is the owner's web UI's (the run it reused still is not listed: ``run``).
            dirs += [Path(d) for r in tree if (d := (r.get("spec") or {}).get("requested_dir"))]
            if self.folder_root is None or any(self._in_folder(d) for d in dirs):
                own += tree
        self.owner = ""
        self._own_jobs = {str(r["id"]) for r in own}
        self._own_runs = {str(i) for r in own for i in r.get("run_ids") or []}
        self._own_dirs = {str(Path(d).resolve()) for r in own for d in r.get("run_dirs") or []}
        self._own_studies = {n for r in own if (n := _study_name(r))}

    def _in_folder(self, path: Path) -> bool:
        if self.folder_root is None:
            return True
        resolved = Path(path).resolve()
        return resolved == self.folder_root or self.folder_root in resolved.parents

    def run(self, run_id: str, name: str, annotations: Mapping[str, str], path: Path) -> bool:
        """Whether a run is the owner's (not a guest's) and lies in the owner's folder."""
        mine = (
            not annotations.get(SUBMITTED_BY)
            or run_id in self._own_runs
            or str(Path(path).resolve()) in self._own_dirs
        )
        return mine and self._in_folder(Path(path))

    def folder(self, relative: str) -> bool:
        """Every workspace folder (the owner's)."""
        return True

    def on_the_way(self, relative: str) -> bool:
        """Every workspace folder (the owner's)."""
        return True

    def plugin(self, registry: str, name: str) -> bool:
        """Every plugin."""
        return True

    def agent(self, name: str) -> bool:
        """Every agent."""
        return True

    def hidden_plugins(self, config: HistoryMatchConfig) -> list[str]:
        """None."""
        return []

    def refused_names(self, config: BaseModel, paths: "PathRules") -> list[str]:
        """None: the owner's files and programs are theirs (a ``[policy]`` still applies)."""
        return []


def _study_name(record: Mapping[str, Any]) -> str | None:
    """Return a study job's study name (from its spec), if it is one."""
    if record.get("kind") != "study":
        return None
    spec = record.get("spec") or {}
    study = (spec.get("study") or {}).get("study") or {}
    name = study.get("name")
    return str(name) if name else None


def _glob_prefix(path: str, pattern: str) -> bool:
    """Whether ``path`` lies under a folder matching ``pattern`` (globs per component)."""
    parts, globs = path.split("/"), pattern.split("/")
    return len(parts) >= len(globs) and all(
        fnmatch.fnmatchcase(p, g) for p, g in zip(parts, globs, strict=False)
    )


#: Text that web-UI users write (the editor and drafts: ``.toml``), or that APEX writes from
#: what they or the model type (reports, run records, logs): never rendered as a template.
WRITTEN_SUFFIXES = (".toml", ".md", ".json", ".jsonl", ".txt", ".log", ".csv")

Where = Literal["own", "draft", "folder", "run", "example"]


class PathRules:
    """Where a UI user's paths lie, for :meth:`Access.refused_names`.

    Parameters
    ----------
    access
        The user's list and work.
    workspace
        The service's workspace.
    own_folder
        The user's web-UI folder, if any.
    entries
        The catalogue's entries (their runs are among them).
    """

    def __init__(
        self,
        access: Access,
        workspace: "Workspace",
        own_folder: Path | None,
        entries: Iterable[Any],
    ) -> None:
        self.access = access
        self.workspace = workspace
        self.own = None if own_folder is None else Path(own_folder).resolve()
        self.entries = list(entries)
        # The web UI's users' folders: only one's own counts, never another's
        # (their drafts, decks and runs), whatever folders the list shows.
        name = workspace.settings.workspace.ui_folder
        roots = workspace.roots
        self.ui_base = (roots[0] / name).resolve() if name and roots else None
        # The catalogue's examples (services/examples.py): the server's files, read-only to
        # every user; a copied example names its deck templates there.
        from apex.services.examples import examples_root

        self.examples = examples_root(workspace.settings.examples)
        if (
            self.examples is not None
            and self.ui_base is not None
            and (
                self.examples == self.ui_base
                or self.examples in self.ui_base.parents
                or self.ui_base in self.examples.parents
            )
        ):
            # Users write in the web UI's folders: a root that holds or lies in them is no
            # longer the server's alone, so it is not trusted.
            logger.warning(
                "the examples root %s overlaps the web UI's folder %s: not trusted",
                self.examples, self.ui_base, extra={"event": "examples_root_untrusted"},
            )  # fmt: skip
            self.examples = None

    def where(self, path: Path) -> Where | None:
        """Return where a path lies, by its real place (a symlink cannot lead out).

        ``draft`` and ``own``: the user's web-UI folder (``drafts/`` in it, or elsewhere);
        ``run``: one of their runs (``access.run_dir``); ``folder``: a visible workspace
        folder; ``example``: the examples root (trusted: shipped with the server, written by
        no user); ``None``: none of them.
        """
        from apex.services.examples import CODE_SUFFIXES
        from apex.services.workspace import WorkspaceError

        target = Path(path).resolve()
        # Only the templates are trusted there (what a copy names in the root); any other
        # file of the root is judged as everywhere else.
        if (self.examples is not None and self.examples in target.parents
                and target.suffix in CODE_SUFFIXES):  # fmt: skip
            return "example"
        if self.own is not None and (target == self.own or self.own in target.parents):
            inner = target.relative_to(self.own).parts
            if inner and inner[0] == "drafts":
                return "draft"
            return "run" if self.access.run_dir(target, self.entries) else "own"
        if self.access.run_dir(target, self.entries):
            return "run"
        if self.ui_base is not None and (target == self.ui_base or self.ui_base in target.parents):
            return None  # another web-UI user's folder
        try:
            relative = self.workspace.relative(target)
        except WorkspaceError:
            return None
        return "folder" if self.access.folder(relative) else None

    def shown(self, value: str) -> str:
        """Return a path as the web UI shows it (workspace-relative), never a server path."""
        from apex.services.workspace import WorkspaceError

        try:
            return self.workspace.relative(Path(value))
        except WorkspaceError:
            return Path(value).name
