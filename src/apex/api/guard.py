"""The REST API's per-request guard.

The owner sees everything; UI users see what ``<root>/ui_access.toml`` allows
(:mod:`apex.services.access`). Hidden things answer ``404`` as if they did not exist;
actions the list does not allow answer ``403``.
"""

import datetime as dt
import functools
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from apex.api.scrub import ROOT_SHOWN, PathScrubber
from apex.core.errors import ApexError
from apex.core.storage import CatalogueEntry
from apex.domain.config import HistoryMatchConfig
from apex.domain.optimization import OptimizationConfig
from apex.services.access import (
    ACCESS_FILE,
    DEFAULT_GUEST,
    SUBMITTED_BY,
    UI_CLIENT,
    Access,
    OwnerView,
    PathRules,
    Role,
    load_access,
    submitter,
)
from apex.services.api import ApexService
from apex.services.identities import UiIdentity, ui_identities
from apex.services.workspace import WorkspaceError

#: How UI users' answers show the service's root (and send it back).
SERVICE_SHOWN = "<service>"


class Denied(ApexError):
    """A request outside what the token may see (404) or do (403)."""

    def __init__(self, status: int, detail: str) -> None:
        super().__init__(detail)
        self.status = status


class Guard:
    """What one request may see and do.

    Parameters
    ----------
    service
        The services.
    role
        ``owner`` (the API token or the owner's web-UI token) or ``ui`` (a guest's UI token).
    guest
        The guest's name (a UI token's).
    expires
        When the request's web-UI token expires; ``None`` for the API token.
    identity
        A kind of web-UI user another package adds (:mod:`apex.services.identities`), whose
        user ``guest`` names: a UI user (``role`` ``ui``) with the kind's own list, and its
        own work in the kind's folder, which the owner's web UI and guests never see.
    """

    def __init__(
        self,
        service: ApexService,
        role: Role,
        guest: str = DEFAULT_GUEST,
        *,
        web_ui: bool = False,
        expires: dt.datetime | None = None,
        identity: UiIdentity | None = None,
    ) -> None:
        self.service = service
        self.role = role
        self.guest = guest
        self.expires = expires
        self.identity = identity if role == "ui" else None
        #: The owner's token from the web UI (``X-APEX-Client: ui``): their own work only.
        self.web_ui = web_ui and role == "owner"
        self._access: Access | None = None
        self._view: Access | None = None

    def again(self) -> "Guard":
        """Return a fresh guard for the same token (its views read again)."""
        return Guard(self.service, self.role, self.guest, web_ui=self.web_ui,
                     expires=self.expires, identity=self.identity)  # fmt: skip

    @property
    def filtered(self) -> bool:
        """Whether this request sees only its own work: a guest's, or the owner's in the UI."""
        return self.ui or self.web_ui

    @property
    def view(self) -> Access:
        """Whose work this request sees: the guest's (their list), or the owner's own."""
        if self.ui:
            return self.access
        if self._view is None:
            # Only the work in the owner's web-UI folder, UI/owner (revision 3); without a UI
            # folder ([workspace] ui_folder = ""), all the owner's work.
            folder = self.service.workspace().ui_user_folder(None, create=False)
            self._view = OwnerView(self.service.jobs.all_records(), folder)
        return self._view

    @property
    def chat_owner(self) -> str:
        """The owner a chat started with this token gets: owner, guest:<name> or <kind>:<name>."""
        return "owner" if not self.ui else self.submitter

    def owns_chat(self, owner: str) -> bool:
        """Whether this token may see a chat of ``owner`` (the owner: all, except in the UI)."""
        if self.web_ui:
            return owner == "owner"
        if not self.ui:
            return True
        # Chats from before named guests belong to the default guest.
        return owner == self.chat_owner or (
            owner == "ui" and self.guest == DEFAULT_GUEST and self.identity is None
        )

    def code(self, given: str | None) -> str | None:
        """Return the code a submission asks for; only the owner may choose it.

        Raises
        ------
        Denied
            When another web-UI user names one: their jobs run the service's default.
        """
        if given is not None and self.ui:
            raise Denied(403, "only the owner chooses the code a job runs")
        return given

    def annotations(self, given: dict[str, str]) -> dict[str, str]:
        """Return a submission's annotations: a UI user's carry who submitted it (server-set)."""
        if not self.ui:
            return given
        return {**given, SUBMITTED_BY: self.submitter}

    @property
    def submitter(self) -> str:
        """A UI user's ``submitted_by``: ``guest:<name>``, or ``<kind>:<name>`` of an added kind."""
        if self.identity is not None:
            return self.identity.submitter(self.guest)
        return submitter(self.guest)

    def own_folder(self, *, create: bool = True) -> Path | None:
        """Return a UI user's web-UI folder (``UI/guests/<name>``, or an added kind's)."""
        space = self.service.workspace()
        if self.identity is not None:
            return space.ui_identity_folder(self.identity.folder, self.guest, create=create)
        return space.ui_user_folder(self.guest if self.ui else None, create=create)

    @property
    def ui(self) -> bool:
        """Whether the request is a UI user's (filtered)."""
        return self.role == "ui"

    @property
    def access(self) -> Access:
        """The list, with the jobs UI users started (read once per request)."""
        if self._access is None:
            records = self.service.jobs.all_records()
            if self.identity is not None:
                self._access = self.identity.access(records, self.guest, self._others_folders())
            else:
                space = self.service.workspace()
                name = space.settings.workspace.ui_folder
                ui_folders = space.relative(space.roots[0] / name) if name and space.roots else None
                self._access = Access(
                    load_access(self.service.root), records, self.guest,
                    ui_folders=ui_folders, own_folder=self._own_ui_folder(),
                )  # fmt: skip
        return self._access

    def _others_folders(self) -> list[str]:
        """Return the folders of the web UI's other users (hidden to an added kind's user)."""
        space = self.service.workspace()
        name = space.settings.workspace.ui_folder
        if name is None or not space.roots:
            return []
        base = space.roots[0] / name
        hidden = [space.relative(base / "owner"), space.relative(base / "guests")]
        for kind in ui_identities():
            users = base / kind.folder
            if users.is_dir():
                hidden += [space.relative(p) for p in users.iterdir() if p.is_dir()
                           and (kind is not self.identity or p.name != self.guest)]  # fmt: skip
        return hidden

    def ui_folder(self, client: str) -> Path | None:
        """Return the web UI's folder for this submission, or ``None``.

        A guest's request always has one (``UI/guests/<name>``); the owner's has one when it
        comes from the web UI, whose client is ``ui`` (``UI/owner``). The client name only
        places the owner's own work, so trusting it gives nothing away.
        """
        if self.ui:
            return self.own_folder()
        if client == UI_CLIENT:
            return self.service.workspace().ui_user_folder(None)
        return None

    def client(self, given: str) -> str:
        """Return the client a submission is recorded under (UI users': ``ui``)."""
        return UI_CLIENT if self.ui else given

    # ------------------------------------------------------------------ reading

    def entries(self) -> list[CatalogueEntry]:
        """Return the catalogue's runs this request may see."""
        entries = list(self.service.store.entries())
        if not self.filtered:
            return entries
        return [e for e in entries if self.view.run(e.run_id, e.name, e.annotations, e.path)]

    def run(self, run_id: str) -> None:
        """Refuse a run this request may not see (as if it did not exist)."""
        if not self.filtered:
            return
        entry = self.service.store.find(run_id)
        if entry is None or not self.view.run(
            entry.run_id, entry.name, entry.annotations, entry.path
        ):
            raise Denied(404, f"no run {run_id!r}")

    def records(self) -> list[dict[str, Any]]:
        """Return the job records this request may see."""
        records = self.service.jobs.all_records()
        if not self.filtered:
            return records
        entries = list(self.service.store.entries())
        return [r for r in records if self.view.job(r, entries)]

    def job(self, job_id: str) -> None:
        """Refuse a job this request may not see (or that does not exist)."""
        try:
            record = self.service.jobs.record(job_id)
        except KeyError:
            raise Denied(404, f"no job {job_id!r}") from None
        if self.filtered and not self.view.job(record, self.service.store.entries()):
            raise Denied(404, f"no job {job_id!r}")

    def visible_jobs(self) -> set[str] | None:
        """Return the ids of the jobs this request may see (``None``: all)."""
        return None if not self.filtered else {str(r["id"]) for r in self.records()}

    def study(self, name: str) -> None:
        """Refuse a study this request may not see."""
        if self.filtered and not self.view.study(name):
            raise Denied(404, f"no study {name!r}")

    def prediction(self, path: Path) -> None:
        """Refuse a prediction that is not under a visible run."""
        if self.filtered and not self.view.run_dir(path, self.service.store.entries()):
            raise Denied(404, f"no prediction at {path}")

    def optimization(self, path: Path) -> None:
        """Refuse an optimization that is not one of this request's visible jobs."""
        if not self.filtered:
            return
        target = Path(path).resolve()
        for record in self.records():
            if record["kind"] in ("optimization", "optimization_resume") and any(
                Path(d).resolve() == target for d in record.get("run_dirs") or []
            ):
                return
        raise Denied(404, f"no optimization at {path}")

    def optimization_config(self, config: OptimizationConfig) -> None:
        """Refuse an optimization that uses plugins UI users may not."""
        if not self.ui:
            return
        from apex.services.plugins import objective_plugins

        used = [
            ("optimizers", config.optimizer.name),
            *((registry, section.name) for registry, section in objective_plugins(config)),
            ("executors", config.execution.name),
        ]
        if config.simulator is not None:
            used.append(("simulators", config.simulator.name))
        used += [("parameterizations", p.name) for p in config.parameterizations]
        hidden = sorted(f"{r}:{n}" for r, n in used if not self.access.plugin(r, n))
        if hidden:
            raise Denied(403, f"plugins not available in the UI: {', '.join(hidden)}")
        self.names(config)

    def plugins(self, infos: Iterable[Any]) -> list[Any]:
        """Return the plugins this request may see."""
        return [p for p in infos if not self.ui or self.access.plugin(p.registry, p.name)]

    def agent(self, name: str) -> bool:
        """Whether this request may see and use an agent."""
        return not self.ui or self.access.agent(name)

    # ------------------------------------------------------------------ the workspace

    def server_path(self, given: str | Path) -> Path:
        """Return the server path of a path a client sent (finding 31, 2026-10-19).

        Absolute paths are taken as they are (checked where they are used, as before);
        ``<service>/...``, as a UI user's answers show the service's root, is under the
        service's root, and ``(workspace)`` is the only root; the other placeholders
        (``<home>``, ``<code>``, ...) are shown only: no guest's work lies under them
        outside the workspace and the service's root, so they are not taken back (404). Other
        relative paths are workspace paths (``<root name>/<rest>``;
        with one root, the path under it), resolved through the workspace. ``..`` is
        refused. A relative path the workspace cannot place is refused for UI users (404)
        and passed on as given for other clients (who send absolute paths: ``Client``).
        """
        path = Path(given)
        if path.is_absolute():
            return path
        if str(given) == ROOT_SHOWN:  # the only root, as free text shows it
            path = Path(".")
        if path.parts and path.parts[0] == SERVICE_SHOWN:
            if ".." in path.parts:
                raise Denied(404, f"{given}: '..' is not allowed in paths")
            return self.service.root.joinpath(*path.parts[1:])
        try:
            return self.service.workspace().absolute(path)
        except WorkspaceError:
            if self.ui:
                raise Denied(404, f"{given} is not in the workspace") from None
            return path

    def optional_path(self, given: str | None) -> Path | None:
        """Return :meth:`server_path` of an optional path (``None`` or empty: ``None``)."""
        return self.server_path(given) if given else None

    def scrubber(self) -> PathScrubber:
        """Return what rewrites this request's server paths as the workspace shows them.

        The workspace's roots (as configured and where they really lie) by their names; the
        service's root as ``<service>`` (and where its links lead, e.g. a ``studies`` link
        into another folder, as ``<service>/<link>``); the frozen copies' ``[jobs]
        code_dir`` as ``<code>``; the machine's resource pools as ``<resources>``; APEX's
        own source as ``<apex>``; Python's virtual environment and its site-packages as
        ``<python>``; the temporary folder as ``<tmp>``; and the home folder as ``<home>``.
        Inner folders win. A placeholder never stands for a system folder (``/usr``,
        ``/opt``, ...: :data:`SYSTEM_FOLDERS`) or one less than two folders deep, which
        would rewrite unrelated paths (``/usr/bin/flow``), except the temporary folder,
        which is ``<tmp>`` even when it is ``/tmp``. Built once per set of folders.
        """
        import sys
        import sysconfig
        import tempfile

        import apex
        from apex.execution.resources import default_root

        space = self.service.workspace()
        several = len(space.roots) > 1
        pairs: list[tuple[str, str]] = []
        configured_roots = [str(r) for r in space.settings.workspace.roots or space.roots]
        for configured, real, name in zip(configured_roots, space.roots, space.names, strict=False):
            shown = name if several else ""
            pairs.append((str(real), shown))
            if Path(configured).is_absolute():  # as configured too (a link to the root)
                pairs.append((configured, shown))
        root = self.service.root
        others: list[tuple[Path, str]] = [(root, SERVICE_SHOWN)]
        try:
            links = [c for c in root.iterdir() if c.is_symlink()]
        except OSError:
            links = []
        others += [(c, f"{SERVICE_SHOWN}/{c.name}") for c in links]
        code = self.service.jobs.settings.jobs.code_dir or root / "code"
        package = Path(apex.__file__).resolve().parent
        others += [
            (Path(code), "<code>"),
            (default_root(), "<resources>"),
            (package.parent, "<apex>"),  # src/ of a checkout, or site-packages
        ]
        if sys.prefix != sys.base_prefix:  # a virtual environment, never the system's
            others.append((Path(sys.prefix), "<python>"))
        others += [(Path(sysconfig.get_paths()[k]), "<python>") for k in ("purelib", "platlib")]
        others.append((Path.home(), "<home>"))
        for folder, shown in others:
            for each in (folder, folder.resolve()):
                if each.is_absolute() and _placeable(each):
                    pairs.append((str(each), shown))
        # The temporary folder is rewritten even when it is a system folder such as /tmp
        # (release tester N1, 2026-10-20): a member's error names its work folder there,
        # and nothing under it is a path a guest could use, so no unrelated path is lost.
        temporary = Path(tempfile.gettempdir())
        for each in (temporary, temporary.resolve()):
            if each.is_absolute() and len(each.parts) >= 2:
                pairs.append((str(each), "<tmp>"))
        return _scrubber(tuple(pairs))

    def relative(self, path: str | Path) -> str:
        """Return a workspace path as the UI shows it (refused outside the workspace)."""
        try:
            return self.service.workspace().relative(path)
        except WorkspaceError:
            if self.ui:
                raise Denied(404, f"{path} is not in the workspace") from None
            raise

    def folder(self, path: str | Path) -> None:
        """Refuse a workspace path outside the visible folders and the guest's own UI folder.

        Both the path as given and where it really lies are judged: a symlink in a visible
        folder (or in their own) that leads into a hidden folder, or into another user's UI
        folder, is refused like the folder it leads to.
        """
        if not self.ui:
            return
        real = self.real(path)
        if real is None:
            raise Denied(404, f"no file or folder {path}")
        for each in (self.relative(path), real):
            if not self.visible(each):
                raise Denied(404, f"no file or folder {path}")

    def real(self, path: str | Path) -> str | None:
        """Return where a workspace path really lies, as the UI shows it (``None``: outside)."""
        space = self.service.workspace()
        try:
            return space.relative(space.real(path))
        except WorkspaceError:
            return None

    def visible(self, relative: str) -> bool:
        """Whether a UI user may open a workspace path.

        Their own web-UI folder, or a listed folder that is not another web-UI user's
        (``Access.folder``).
        """
        return self.own_ui(relative) or self.access.folder(relative)

    def _own_ui_folder(self) -> str | None:
        """Return the guest's web-UI folder as the workspace shows it, if any."""
        space = self.service.workspace()
        if self.identity is not None:  # its own, with or without a guests' list
            folder = space.ui_identity_folder(self.identity.folder, self.guest, create=False)
            return None if folder is None else space.relative(folder)
        name = space.settings.workspace.ui_folder
        # Without a list a guest sees nothing, not even their own folder.
        listed = (self.service.root / ACCESS_FILE).exists()
        if not self.ui or name is None or not space.roots or not listed:
            return None
        return space.relative(space.roots[0] / name / "guests" / self.guest)

    def own_ui(self, relative: str) -> bool:
        """Whether a workspace path is in this guest's own web-UI folder (their drafts...)."""
        own = self._own_ui_folder()
        relative = relative.strip("/")
        return own is not None and (relative == own or relative.startswith(own + "/"))

    def listed(self, entry_path: str, kind: str) -> bool:
        """Whether a workspace listing shows an entry (visible, or a folder leading to one)."""
        if not self.ui:
            return True
        real = self.real(entry_path)
        if real is None:
            return False
        if real != entry_path.strip("/"):  # a link: shown only if what it leads to is
            return self.visible(entry_path) and self.visible(real)
        return self.visible(entry_path) or (
            kind == "folder" and (self.access.on_the_way(entry_path) or self._to_own_ui(entry_path))
        )

    def on_the_way(self, relative: str) -> bool:
        """Whether a folder leads to a visible one, or to this guest's web-UI folder."""
        return self.access.on_the_way(relative) or self._to_own_ui(relative)

    def own_ui_draft(self, relative: str) -> bool:
        """Whether a workspace path is in the drafts of this guest's web-UI folder."""
        own = self._own_ui_folder()
        return own is not None and relative.strip("/").startswith(own + "/drafts/")

    def _to_own_ui(self, entry_path: str) -> bool:
        """Whether a folder leads to this guest's web-UI folder (``UI``, ``UI/guests``)."""
        own = self._own_ui_folder()
        return own is not None and own.startswith(entry_path.strip("/") + "/")

    # ------------------------------------------------------------------ doing

    def action(self, name: str) -> None:
        """Refuse an action (submit, stop, resume, edit) the list does not allow."""
        if self.ui and not getattr(self.access.rules.actions, name):
            raise Denied(403, f"UI users may not {name} ([actions] {name} in ui_access.toml)")

    def config(self, config: HistoryMatchConfig) -> None:
        """Refuse a configuration with plugins, files or programs UI users may not use.

        Files and programs: ``Access.refused_names``.
        """
        if not self.ui:
            return
        hidden = self.access.hidden_plugins(config)
        if hidden:
            raise Denied(403, f"plugins not available in the UI: {', '.join(hidden)}")
        self.names(config)

    def names(self, config: BaseModel) -> None:
        """Refuse a configuration that names files or programs a UI user may not."""
        if not self.ui:
            return
        refused = self.access.refused_names(config, self.paths())
        if refused:
            raise Denied(403, "not available in the UI: " + "; ".join(refused))

    def paths(self) -> PathRules:
        """Return where this UI user's paths may lie (``Access.refused_names``)."""
        space = self.service.workspace()
        own: Path | None
        if self.identity is not None:
            own = space.ui_identity_folder(self.identity.folder, self.guest, create=False)
        elif (self.service.root / ACCESS_FILE).exists() and space.roots:
            own = space.ui_user_folder(self.guest, create=False)
        else:
            own = None
        return PathRules(self.access, space, own, self.service.store.entries())

    def file(self, path: Path) -> None:
        """Refuse a file (a configuration, study or prediction) outside the visible folders."""
        if self.ui:
            self.folder(path)


#: Folders a placeholder never stands for: rewriting them would rewrite unrelated paths.
SYSTEM_FOLDERS = frozenset(
    Path(p) for p in ("/", "/usr", "/usr/local", "/opt", "/tmp", "/home", "/root", "/var")
)


def _placeable(folder: Path) -> bool:
    """Whether a placeholder may stand for a folder: two folders deep, not a system one."""
    return folder not in SYSTEM_FOLDERS and len(folder.parts) >= 3


@functools.lru_cache(maxsize=32)
def _scrubber(pairs: tuple[tuple[str, str], ...]) -> PathScrubber:
    """Return the scrubber of a set of folders (its patterns compiled once)."""
    return PathScrubber(pairs)
