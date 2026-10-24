"""How long, and how much, the web UI's guests' work is kept.

Only work a guest submitted is pruned: jobs annotated ``submitted_by = guest:<name>``,
which the server sets on every guest submission. The owner's work, and other
local programs' (the CLI, the Python client), never is.

The unit is a guest's **submission**: a top-level job with its children (a study's cells),
the run directories the guest's jobs created (their catalogue entries carry the guest's
annotation), prediction and optimization outputs (a resumed optimization's job
shares its directory), and a study's record. A run a guest merely *reused* (an
identical configuration of someone else's) belongs to its creator and stays.

:func:`prune_guests` applies ``[guests]``:
- a finished submission older than ``retention_days`` (30) is deleted;
- then, while a guest's submissions take more than ``max_storage_mb`` (500) on disk, their
  oldest finished one is deleted.

A copy of the examples catalogue (``UI/<user>/examples/<id>``, found by its marker file;
revision 2) is a unit too, dated when it was copied: kept for ``retention_days`` and counted
against ``max_storage_mb`` as a submission is. A run started from a copy writes inside it and
is a submission of its own: the copy's size leaves it out, and the copy stays until it has
gone. So does a copy that a queued or running job, or a catalogued run, names (a web-UI
run lies in the user's ``runs/`` folder, but reads the copy's files).

What a deletion empties goes too, up to the guest's own web-UI folder
(:func:`apex.services.deletion.remove_empty_parents`). A submission that is kept (see
below) still counts against ``max_storage_mb``, and a deletion counts only the bytes it
really freed.

Submissions that share a directory (an optimization and its resumptions, or the same
optimization submitted twice) are one unit: deleted together, active while any is, as old
as the newest, and their directory counted once. Queued and running jobs are never
deleted, nor a directory a process holds locked (``apex optimize --resume`` from the
command line), and no directory is deleted outside the service root and the workspace
roots.
"""

import datetime as dt
import json
import logging
import re
import shutil
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from apex.services.access import SUBMITTED_BY
from apex.services.api import ApexService
from apex.services.deletion import remove_empty_parents
from apex.services.identities import identity_of
from apex.services.jobs import FINAL
from apex.services.settings import GuestsSection
from apex.storage.locks import locked

logger = logging.getLogger("apex.service")

#: Jobs whose outputs are their ``run_dirs`` (not catalogued runs).
OUTPUT_KINDS = ("prediction", "optimization", "optimization_resume")


@dataclass
class Submission:
    """One guest submission and what it holds on disk."""

    guest: str
    job: str
    created: dt.datetime
    jobs: list[str] = field(default_factory=list)
    active: bool = False
    run_ids: list[str] = field(default_factory=list)
    dirs: list[Path] = field(default_factory=list)
    studies: list[str] = field(default_factory=list)
    size: int = 0


def _parse(stamp: str) -> dt.datetime:
    value = dt.datetime.fromisoformat(stamp)
    return value if value.tzinfo else value.replace(tzinfo=dt.UTC)


def _size(path: Path) -> int:
    if path.is_symlink():
        return 0  # its target is not the guest's, and unlinking frees nothing
    if path.is_file():
        return path.stat().st_size
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file() and not p.is_symlink())


def _inside(path: Path, bases: Iterable[Path]) -> bool:
    resolved = path.resolve()
    return any(resolved == b or b in resolved.parents for b in bases)


def submissions(
    service: ApexService, prefix: str = "guest:", *, exact: bool = False
) -> list[Submission]:
    """Return every guest submission, oldest first.

    With ``prefix``, those whose submitter starts with it; with ``exact``, is it.
    """
    records = service.jobs.all_records()
    children: dict[str, list[Mapping[str, Any]]] = {}
    for record in records:
        if record.get("parent"):
            children.setdefault(str(record["parent"]), []).append(record)
    entries = {e.run_id: e for e in service.store.entries()}
    found: list[Submission] = []
    for record in records:
        owner = (record.get("annotations") or {}).get(SUBMITTED_BY, "")
        matches = owner == prefix if exact else owner.startswith(prefix)
        if not matches or record.get("parent"):
            continue
        unit = Submission(owner, str(record["id"]), _parse(str(record["created"])))
        for job in [record, *children.get(str(record["id"]), [])]:
            unit.jobs.append(str(job["id"]))
            unit.active |= job["state"] not in FINAL
            for run_id in job.get("run_ids") or []:
                entry = entries.get(run_id)
                # Only runs the guest created: a reused run carries its creator's annotation.
                if entry is not None and entry.annotations.get(SUBMITTED_BY) == owner:
                    unit.run_ids.append(run_id)
                    unit.dirs.append(entry.path)
            if job.get("kind") in OUTPUT_KINDS:
                unit.dirs += [Path(d) for d in job.get("run_dirs") or []]
            if job.get("kind") == "study":
                name = (((job.get("spec") or {}).get("study") or {}).get("study") or {}).get("name")
                if name:
                    unit.studies.append(str(name))
        unit.size = sum(_size(d) for d in unit.dirs if d.exists())
        found.append(unit)
    return sorted(found, key=lambda u: u.created)


def delete(service: ApexService, unit: Submission) -> tuple[bool, int, bool]:
    """Delete a submission: its directories (inside the allowed bases), entries and jobs.

    All or nothing for the records: if a directory stays (outside the bases, locked, a
    failed removal), the entries and jobs stay too, so that nothing is left on disk without
    a record that leads to it; the next pruning tries again. A symlinked directory loses
    its link only: the files it points to are not the guest's to delete. Returns whether
    the submission was deleted, the bytes freed on disk (a partly removed directory's share
    too), and whether a removal failed (what it left then still counts against the limit;
    a submission kept because it is in use or outside the bases does not).
    """
    bases = [service.root.resolve(), *service.workspace().roots]
    store = service.jobs.store
    kept = False
    failed = False
    freed = 0
    for path in unit.dirs:
        if path.is_symlink():
            if _inside(path.parent, bases):  # the link's place, not its target's
                path.unlink()
            else:
                kept = True
            continue
        if not path.exists():
            continue
        if not _inside(path, bases):
            logger.warning(
                "guest retention: %s is outside the service root and the workspace; kept",
                path, extra={"event": "guest_prune_skipped", "path": str(path)},
            )  # fmt: skip
            kept = True
            continue
        if locked(path):
            logger.warning(
                "guest retention: %s is in use by a running process; kept", path,
                extra={"event": "guest_prune_skipped", "path": str(path)},
            )  # fmt: skip
            kept = True
            continue
        before = _size(path)
        shutil.rmtree(path, ignore_errors=True)
        freed += before - (_size(path) if path.exists() else 0)
        if path.exists():
            logger.warning(
                "guest retention: %s could not be removed entirely; its records are kept",
                path, extra={"event": "guest_prune_failed", "path": str(path)},
            )  # fmt: skip
            kept = failed = True
            continue
        remove_empty_parents(path.parent, service)
    if kept:
        return False, freed, failed
    for run_id in unit.run_ids:
        store.forget_run(run_id)
    for name in unit.studies:
        store.delete_study(name)
        # Its study.json and comparison, which status and comparisons read by name.
        folder = service.root / "studies" / name
        if folder.is_dir() and Path(name).name == name:
            shutil.rmtree(folder, ignore_errors=True)
    for job in unit.jobs:
        store.delete_job(job)
    return True, freed, False


def merge_shared(units: list[Submission]) -> list[Submission]:
    """Merge submissions that share a directory into one unit (see the module docstring).

    Connected components: a resumption of a resumption chains. The merged unit is named by
    its oldest job, dated by its newest, active if any is, and sized over its distinct
    directories. Oldest first, by that date.
    """
    parent = list(range(len(units)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    owner: dict[Path, int] = {}
    for i, unit in enumerate(units):
        for path in unit.dirs:
            key = path.resolve()
            if key in owner:
                parent[find(i)] = find(owner[key])
            else:
                owner[key] = i
    groups: dict[int, list[Submission]] = {}
    for i, unit in enumerate(units):
        groups.setdefault(find(i), []).append(unit)
    merged = []
    for members in groups.values():
        if len(members) == 1:
            merged.append(members[0])
            continue
        first = min(members, key=lambda u: u.created)
        dirs = list(dict.fromkeys(d for u in members for d in u.dirs))
        unit = Submission(
            first.guest, first.job, max(u.created for u in members),
            jobs=[j for u in members for j in u.jobs],
            active=any(u.active for u in members),
            run_ids=list(dict.fromkeys(r for u in members for r in u.run_ids)),
            dirs=dirs,
            studies=list(dict.fromkeys(n for u in members for n in u.studies)),
            size=sum(_size(d) for d in dirs if d.exists()),
        )  # fmt: skip
        merged.append(unit)
    return sorted(merged, key=lambda u: u.created)


def example_copies(
    service: ApexService, prefix: str, others: list[Submission], *, exact: bool = False
) -> list[Submission]:
    """Return the example copies in the web-UI folders of ``prefix``'s users, as units.

    ``others``: the users' submissions. A copy holding one of their directories, or a
    catalogued run, is active and its size leaves them out. A copy named by a queued or
    running job, or by a catalogued run's configuration (a run started from it, wherever its
    directory is), is active too: none is deleted while something may still read it.
    """
    from apex.services.examples import COPY_MARKER

    space = service.workspace()
    folders: list[tuple[str, Path]] = []
    added = identity_of(prefix) if exact else None
    if added is not None:
        folder = space.ui_identity_folder(added[0].folder, added[1], create=False)
        folders = [] if folder is None else [(prefix, folder)]
    elif exact:
        folder = space.ui_user_folder(prefix.split(":", 1)[1], create=False)
        folders = [] if folder is None else [(prefix, folder)]
    else:
        probe = space.ui_user_folder("guest", create=False)
        guests = None if probe is None else probe.parent
        if guests is not None and guests.is_dir():
            folders = [(f"guest:{d.name}", d) for d in sorted(guests.iterdir()) if d.is_dir()]
    # The submissions' directories and every catalogued run: a copy holding a run that some
    # record still names is not deleted under it (conservative: an orphaned run keeps it).
    entries = service.store.entries()
    paths = [d for unit in others for d in unit.dirs] + [e.path for e in entries]
    held = sorted({d.resolve() for d in paths if d.exists()})
    # What names a copy's files: every job record (its configuration's path and spec) and
    # every catalogued run's resolved configuration (its case_dir, observations, ...). A
    # web-UI run is placed in the user's runs/ folder, not in the copy, yet reads the copy.
    # Jobs still queued or running only: a finished one's run is catalogued (below), and its
    # record alone would keep the copy until the record itself is pruned.
    references = [json.dumps(r, default=str, ensure_ascii=False)
                  for r in service.jobs.all_records() if r.get("state") not in FINAL]  # fmt: skip
    for entry in entries:
        try:
            references.append((entry.path / "config.json").read_text("utf-8"))
        except OSError:
            continue
    found: list[Submission] = []
    for submitter, folder in folders:
        examples = folder / "examples"
        if not examples.is_dir() or examples.is_symlink():
            continue
        for marker in sorted(examples.rglob(COPY_MARKER)):
            copy = marker.parent
            try:
                stamp = json.loads(marker.read_text("utf-8")).get("created", "")
                created = _parse(str(stamp))
            except (OSError, ValueError, AttributeError):
                created = dt.datetime.fromtimestamp(marker.stat().st_mtime, dt.UTC)
            inside = [d for d in held if copy.resolve() in d.parents]
            size = max(0, _size(copy) - sum(_size(d) for d in inside))
            # The path, then a separator or its string's end: a quote, an escaped one (a
            # configuration held as JSON text inside a record), or the text's end.
            named = re.compile(re.escape(str(copy.resolve())) + r'(?:/|\\*"|$)')
            used = any(named.search(text) for text in references)
            found.append(Submission(
                submitter, f"example:{copy.relative_to(folder).as_posix()}", created,
                active=bool(inside) or used, dirs=[copy], size=size,
            ))  # fmt: skip
    return found


def prune_guests(
    service: ApexService, settings: GuestsSection | None = None, now: dt.datetime | None = None
) -> list[Submission]:
    """Apply ``[guests]`` (see the module docstring); return what was deleted."""
    rules = settings or service.jobs.settings.guests
    return _prune(service, "guest:", rules.retention_days, rules.max_storage_mb, now)


def prune_user(
    service: ApexService,
    user: str,
    retention_days: float,
    max_storage_mb: float,
    now: dt.datetime | None = None,
) -> list[Submission]:
    """Prune one web-UI user's work, as a guest's, by its limits.

    Raises
    ------
    ValueError
        Unless ``user`` is ``guest:<name>`` or a user of a kind another package adds.
    """
    if identity_of(user) is None and not (user.startswith("guest:") and len(user) > 6):
        raise ValueError(f"{user!r} is no web-UI user's: guest:<name> or <kind>:<name>")
    return _prune(service, user, retention_days, max_storage_mb, now, exact=True)


def _prune(
    service: ApexService,
    prefix: str,
    retention_days: float,
    max_storage_mb: float,
    now: dt.datetime | None,
    *,
    exact: bool = False,
) -> list[Submission]:
    moment = now or dt.datetime.now(dt.UTC)
    cutoff = moment - dt.timedelta(days=retention_days)
    limit = int(max_storage_mb * 1e6)
    deleted: list[Submission] = []
    by_guest: dict[str, list[Submission]] = {}
    units = merge_shared(submissions(service, prefix, exact=exact))
    units = sorted([*units, *example_copies(service, prefix, units, exact=exact)],
                   key=lambda u: u.created)  # fmt: skip
    tried: set[str] = set()
    for unit in units:
        if not unit.active and unit.created < cutoff:
            tried.add(unit.job)
            done, freed, failed = delete(service, unit)
            if done:
                unit.size = freed  # logged below: what was really freed
                deleted.append(unit)
                continue
            if not failed:
                # In use (a locked directory) or outside the bases: not counted, so that a
                # run being predicted from does not push the guest's newer work out.
                continue
            # A failed removal: what is left of it still takes the guest's space.
            unit.size = max(0, unit.size - freed)
        by_guest.setdefault(unit.guest, []).append(unit)
    for units in by_guest.values():
        total = sum(u.size for u in units)
        for unit in units:  # oldest first
            if total <= limit:
                break
            if unit.active or unit.job in tried:
                continue
            tried.add(unit.job)
            done, freed, _ = delete(service, unit)
            total -= freed  # what was really freed, not the unit's size
            if done:
                unit.size = freed
                deleted.append(unit)
    for unit in deleted:
        logger.info(
            "guest retention: deleted %s's job %s (%d bytes)", unit.guest, unit.job, unit.size,
            extra={"event": "guest_pruned", "guest": unit.guest, "job": unit.job,
                   "bytes": unit.size},
        )  # fmt: skip
    return deleted
