"""Deleting one's own work from the web UI.

The unit is a **submission**, as for the guests' retention: a top-level job with
its children (a study's cells), the run directories it *created*, their predictions, and a
study's record. A person may delete only their own: a guest the jobs annotated
``submitted_by = guest:<name>``, the owner those without that annotation.

What a submission takes with it, and what it leaves:

- **Runs it created.** A run's creator is the first job that used it (it registered the
  run); reuses come later, and a crash recovery that turns the creating job into a resume
  keeps its place. A run another submission also uses is kept (*kept*), except this
  person's own later submissions that only resume or reuse it, which go with it.
- **Predictions.** Those written in its runs go with them if they are this person's;
  another person's prediction there refuses the deletion. A prediction's own folder is
  deleted only inside one of this person's runs or their web-UI folder, and only if no
  other submission writes there.
- **Folders.** Each must lie inside the workspace, not be a workspace root or a symbolic
  link, hold no other run, lie in no other person's run, and contain only what APEX writes
  there (:data:`RUN_ENTRIES`, :data:`PREDICTION_ENTRIES`). A run's or prediction's
  ``output_dir`` is chosen by people, so a folder that fails this is refused, not trusted.
- **Studies.** The record (``<root>/studies/<name>/``: ``study.json``, the comparison).

A submission with anything refused, or still queued or running, is not deleted at all.
:func:`plan` says what a deletion would remove, with a ``token`` over it; :func:`delete`
needs that token, and plans again and removes under the store's lock, so what is removed is
exactly what the person confirmed.
"""

import fnmatch
import hashlib
import json
import logging
import shutil
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from apex.core.errors import ApexError
from apex.core.storage import CatalogueEntry
from apex.services.access import SUBMITTED_BY
from apex.services.api import ApexService
from apex.services.identities import identity_of
from apex.services.jobs import FINAL

logger = logging.getLogger("apex.service")

#: The owner's work: jobs without a ``submitted_by`` annotation.
OWNER = "owner"
#: What APEX writes at the top of a run's folder (``storage/run_directory.py``), with what
#: reports, QA/QC and predictions add there.
RUN_ENTRIES = (
    "config.json", "state.json", "history.json", "summary.json", "events.jsonl", "run.lock",
    "provenance.json", "snapshot_version", "stop_requested", "checkpoints", "simulations",
    "agents", "predictions", "report", "qaqc",
)  # fmt: skip
#: What APEX writes at the top of an optimization's folder (``storage/optimization_directory``).
OPTIMIZATION_ENTRIES = (
    "config.json", "history.jsonl", "checkpoint.npz", "checkpoint.json", "*.tmp",
    "result.json", "evaluations.jsonl", "simulations", "run.lock", "stop_requested",
    "agents", "report",
)  # fmt: skip
#: Jobs whose ``run_dirs`` are optimizations' folders.
OPTIMIZATION_KINDS = ("optimization", "optimization_resume")
#: What APEX writes at the top of a prediction's folder (``storage/prediction_directory.py``).
PREDICTION_ENTRIES = (
    "config.json", "provenance.json", "state.json", "members", "forecast_*.npz", "truth.npz",
    "statistics.csv", "scores.json", "scores.csv", "events.jsonl", "simulations", "report",
    "run.lock",
)  # fmt: skip


class DeletionRefused(ApexError):
    """A deletion that may not happen as asked (see :attr:`DeletionPlan.blocked`)."""


@dataclass
class DeletionPlan:
    """What deleting one submission removes."""

    job: str
    kind: str
    owner: str
    jobs: list[str] = field(default_factory=list)
    runs: list[str] = field(default_factory=list)
    dirs: list[Path] = field(default_factory=list)
    studies: list[str] = field(default_factory=list)
    kept: list[str] = field(default_factory=list)
    size: int = 0
    blocked: list[str] = field(default_factory=list)
    #: Of ``dirs``, the optimizations' folders (their entries differ from a run's).
    optimizations: list[Path] = field(default_factory=list)

    @property
    def token(self) -> str:
        """A digest of what would be removed: :func:`delete` must get it back."""
        content = {
            "jobs": sorted(self.jobs), "runs": sorted(self.runs),
            "dirs": sorted(str(d) for d in self.dirs), "studies": sorted(self.studies),
        }  # fmt: skip
        return hashlib.sha256(json.dumps(content, sort_keys=True).encode()).hexdigest()[:32]

    def document(self) -> dict[str, Any]:
        """Return the plan as JSON (paths as given; the API shows them relative)."""
        data = asdict(self)
        data["dirs"] = [str(d) for d in self.dirs]
        data["token"] = self.token
        return data


def owner_of(annotations: Mapping[str, str] | None) -> str:
    """Return who submitted a job or created a run: ``guest:<name>`` or :data:`OWNER`."""
    return (annotations or {}).get(SUBMITTED_BY) or OWNER


def top_level(service: ApexService, job_id: str) -> dict[str, Any]:
    """Return the submission a job belongs to (its top-level job).

    Raises
    ------
    KeyError
        If there is no such job.
    """
    record = service.jobs.record(job_id)
    while record.get("parent"):
        record = service.jobs.record(str(record["parent"]))
    return record


def job_of_run(service: ApexService, run_id: str, owner: str) -> str:
    """Return ``owner``'s submission of a run: the one that created it, else their latest.

    Raises
    ------
    KeyError
        If no submission of ``owner`` used the run.
    """
    records = service.jobs.all_records()
    by_id = {str(r["id"]): r for r in records}
    using = [_root(r, by_id) for r in records if run_id in (r.get("run_ids") or [])]
    mine = [j for j in using if owner_of(by_id[j].get("annotations")) == owner]
    if not mine:
        raise KeyError(f"no submission of yours ran {run_id!r}")
    return mine[0] if mine[0] == using[0] else mine[-1]


def job_of_prediction(service: ApexService, path: Path, owner: str) -> str:
    """Return ``owner``'s latest submission of the prediction written to ``path``.

    Raises
    ------
    KeyError
        If no prediction job of ``owner`` wrote there.
    """
    records = service.jobs.all_records()
    by_id = {str(r["id"]): r for r in records}
    target = Path(path).resolve()
    for record in reversed(records):
        dirs = [Path(d).resolve() for d in record.get("run_dirs") or []]
        top = _root(record, by_id)
        if (
            record.get("kind") == "prediction"
            and target in dirs
            and owner_of(by_id[top].get("annotations")) == owner
        ):
            return top
    raise KeyError(f"no prediction of yours at {path}")


def job_of_study(service: ApexService, name: str) -> str:
    """Return a study's job.

    Raises
    ------
    KeyError
        If there is no such study.
    """
    from apex.services.study import study_record

    try:
        return str(study_record(service, name)["job"])
    except ApexError as error:
        raise KeyError(f"no study {name!r}") from error


def plan(service: ApexService, job_id: str) -> DeletionPlan:
    """Return what deleting the submission of ``job_id`` would remove (nothing is deleted).

    Raises
    ------
    KeyError
        If there is no such job.
    """
    records = service.jobs.all_records()
    by_id = {str(r["id"]): r for r in records}
    head = top_level(service, job_id)
    owner = owner_of(head.get("annotations"))
    unit = DeletionPlan(str(head["id"]), str(head.get("kind")), owner)
    entries = {e.run_id: e for e in service.store.entries()}

    def tree(top: str) -> set[str]:
        return {top} | {i for i in by_id if _descends(by_id[i], top, by_id)}

    def submitter(record: Mapping[str, Any]) -> str:
        return owner_of(by_id[_root(record, by_id)].get("annotations"))

    members = tree(unit.job)
    creator: dict[str, str] = {}  # the first job that used a run registered it
    for record in records:
        for run_id in record.get("run_ids") or []:
            creator.setdefault(run_id, str(record["id"]))
    created = {
        run_id for run_id, job in creator.items()
        if job in members and run_id in entries and owner_of(entries[run_id].annotations) == owner
    }  # fmt: skip
    # This person's later submissions that only resume or reuse those runs go with them.
    for record in records:
        runs = set(record.get("run_ids") or [])
        top = _root(record, by_id)
        if top not in members and runs and runs <= created and submitter(record) == owner:
            members |= tree(top)
    # An optimization's folder is its run: this person's other submissions that
    # use it (a resumption, the same optimization again) go with it, as resumed runs do.
    # Until nothing more joins: a joined submission may bring folders of its own.
    shared_with_others = False
    while True:
        optimizing = {
            Path(d).resolve() for i in members if by_id[i].get("kind") in OPTIMIZATION_KINDS
            for d in by_id[i].get("run_dirs") or []
        }  # fmt: skip
        joined = False
        for record in records:
            top = _root(record, by_id)
            uses = {Path(d).resolve() for d in record.get("run_dirs") or []} & optimizing
            if top in members or not uses or record.get("kind") not in OPTIMIZATION_KINDS:
                continue
            if submitter(record) == owner:
                members |= tree(top)
                joined = True
            else:
                shared_with_others = True
        if not joined:
            break
    if shared_with_others:
        unit.blocked.append("another person's optimization uses its folder")
    elsewhere = {
        run for i, r in by_id.items() if i not in members for run in r.get("run_ids") or []
    }
    unit.runs = sorted(created - elsewhere)
    unit.kept = sorted(created & elsewhere)
    run_dirs = [Path(entries[r].path) for r in unit.runs]
    # Predictions written inside the runs that go: this person's go too; others' refuse.
    for record in records:
        if record.get("kind") != "prediction" or str(record["id"]) in members:
            continue
        outputs = [Path(d) for d in record.get("run_dirs") or []]
        if any(_within(o, d) for o in outputs for d in run_dirs):
            if submitter(record) == owner:
                members |= tree(_root(record, by_id))
            else:
                unit.blocked.append("another person's prediction is written in one of its runs")
    unit.dirs = list(run_dirs)
    own_runs = [Path(e.path) for e in entries.values() if owner_of(e.annotations) == owner]
    added = identity_of(owner)  # a user of a kind another package adds
    if added is not None:
        ui_folder = service.workspace().ui_identity_folder(added[0].folder, added[1])
    else:
        guest = None if owner == OWNER else owner.removeprefix("guest:")
        ui_folder = service.workspace().ui_user_folder(guest)
    for job in sorted(members, key=lambda i: str(by_id[i].get("created", ""))):
        record = by_id[job]
        unit.jobs.append(job)
        if record["state"] not in FINAL:
            unit.blocked.append(f"job {job} is {record['state']}: stop it first")
        if record.get("kind") == "prediction":
            for output in (Path(d) for d in record.get("run_dirs") or []):
                if any(_within(output, d) for d in run_dirs):
                    continue  # goes with its run
                shared = any(
                    _within(output, Path(d)) and _within(Path(d), output)
                    for i, r in by_id.items() if i not in members for d in r.get("run_dirs") or []
                )  # fmt: skip
                mine = any(_within(output, d) for d in own_runs) or (
                    ui_folder is not None and _within(output, ui_folder)
                )
                if shared or not mine:
                    why = "another submission writes there" if shared else "not in your runs"
                    unit.blocked.append(f"prediction folder {output.name}: {why}")
                else:
                    unit.dirs.append(output)
        if record.get("kind") in OPTIMIZATION_KINDS:
            for output in (Path(d) for d in record.get("run_dirs") or []):
                if output.resolve() not in {d.resolve() for d in unit.dirs}:
                    unit.dirs.append(output)
                    unit.optimizations.append(output)
        if record.get("kind") == "study":
            study = ((record.get("spec") or {}).get("study") or {}).get("study") or {}
            if study.get("name"):
                unit.studies.append(str(study["name"]))
    _check_folders(service, unit, entries)
    unit.size = sum(_size(d) for d in unit.dirs if d.is_dir() and not d.is_symlink())
    return unit


def _check_folders(
    service: ApexService, unit: DeletionPlan, entries: Mapping[str, CatalogueEntry]
) -> None:
    """Refuse folders outside the workspace, or holding anything but APEX's output."""
    roots = [r.resolve() for r in service.workspace().roots]
    outside = [d for d in unit.dirs if not _inside(d, roots)]
    if outside:
        unit.blocked.append(
            "outside the workspace, so not deleted from the web UI: "
            + ", ".join(d.name for d in outside[:5])  # names: no server path for the UI
        )
    going = set(unit.runs)
    others = [Path(e.path).resolve() for e in entries.values() if e.run_id not in going]
    foreign = [
        Path(e.path).resolve() for e in entries.values()
        if e.run_id not in going and owner_of(e.annotations) != unit.owner
    ]  # fmt: skip
    for index, folder in enumerate(unit.dirs):
        if not _inside(folder, roots) or not (folder.exists() or folder.is_symlink()):
            continue
        allowed = (
            RUN_ENTRIES if index < len(unit.runs)
            else OPTIMIZATION_ENTRIES if folder in unit.optimizations else PREDICTION_ENTRIES
        )  # fmt: skip
        reason = _unsafe(folder, roots, others, foreign, allowed)
        if reason:
            unit.blocked.append(f"{folder.name}: {reason}")


def _unsafe(
    folder: Path,
    roots: Sequence[Path],
    others: Sequence[Path],
    foreign: Sequence[Path],
    allowed: Sequence[str],
) -> str | None:
    """Return why a folder may not be deleted, or ``None`` (the module docstring)."""
    if folder.is_symlink() or not folder.is_dir():
        return "not a folder (a link or a file)"
    resolved = folder.resolve()
    if resolved in roots:
        return "a workspace root"
    if any(o == resolved or resolved in o.parents for o in others):
        return "another run is inside it"
    if any(o in resolved.parents for o in foreign):
        return "it lies in another person's run"
    unknown = sorted(
        p.name for p in resolved.iterdir()
        if not any(fnmatch.fnmatchcase(p.name, pattern) for pattern in allowed)
    )  # fmt: skip
    if unknown:
        return "not only APEX output (" + ", ".join(unknown[:5]) + ")"
    return None


def delete(service: ApexService, job_id: str, token: str, owner: str) -> DeletionPlan:
    """Delete the submission of ``job_id`` as :func:`plan` described it under ``token``.

    Raises
    ------
    DeletionRefused
        If it is not ``owner``'s, is blocked, changed since it was planned, or a folder
        could not be removed (then nothing else of it is forgotten).
    KeyError
        If there is no such job.
    """
    with service.jobs.store.locked():  # nothing is queued or reuses a run meanwhile
        unit = plan(service, job_id)
        if unit.owner != owner:
            raise DeletionRefused(f"job {unit.job} is not yours to delete")
        if unit.blocked:
            raise DeletionRefused("; ".join(unit.blocked))
        if token != unit.token:
            raise DeletionRefused(
                "what this deletes has changed since you confirmed it: look again"
            )
        remove(service, unit)
    logger.info(
        "deleted %s's job %s (%d bytes)", unit.owner, unit.job, unit.size,
        extra={"event": "submission_deleted", "owner": unit.owner, "job": unit.job,
               "bytes": unit.size},
    )  # fmt: skip
    return unit


def remove(service: ApexService, unit: DeletionPlan) -> None:
    """Remove a planned submission's folders, then its entries, study records and jobs.

    Raises
    ------
    DeletionRefused
        If a folder is still there afterwards (nothing else is forgotten then).
    """
    left = []
    for path in unit.dirs:
        if path.is_dir() and not path.is_symlink():
            resolved = path.resolve()
            shutil.rmtree(resolved, ignore_errors=True)
            remove_empty_parents(resolved.parent, service)
        if path.exists() or path.is_symlink():
            left.append(path.name)
    if left:
        raise DeletionRefused(f"could not delete {', '.join(left)}; nothing else was removed")
    store = service.jobs.store
    for run_id in unit.runs:
        store.forget_run(run_id)
    for name in unit.studies:
        store.delete_study(name)
        folder = service.root / "studies" / name
        if folder.is_dir() and Path(name).name == name:
            shutil.rmtree(folder, ignore_errors=True)
    for job in unit.jobs:
        store.delete_job(job)
        folder = service.root / "jobs" / job  # its worker's log
        if folder.is_dir() and Path(job).name == job:
            shutil.rmtree(folder, ignore_errors=True)


def _root(record: Mapping[str, Any], by_id: Mapping[str, Mapping[str, Any]]) -> str:
    current = record
    while current.get("parent") and str(current["parent"]) in by_id:
        current = by_id[str(current["parent"])]
    return str(current["id"])


def _descends(record: Mapping[str, Any], head: str, by_id: Mapping[str, Mapping[str, Any]]) -> bool:
    parent = record.get("parent")
    while parent:
        if parent == head:
            return True
        parent = by_id.get(str(parent), {}).get("parent")
    return False


def _within(path: Path, folder: Path) -> bool:
    resolved, base = path.resolve(), folder.resolve()
    return resolved == base or base in resolved.parents


def _inside(path: Path, roots: Sequence[Path]) -> bool:
    return any(_within(path, r) for r in roots)


def _size(path: Path) -> int:
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file() and not p.is_symlink())


def remove_empty_parents(folder: Path, service: ApexService) -> None:
    """Remove the empty folders deleted work leaves, from ``folder`` upwards.

    Inside a web-UI user's folder (``UI/owner``, ``UI/guests/<name>``,
    an added kind's) up to it, not including it: an emptied ``runs/`` or ``studies/``
    goes, the user's folder stays. Elsewhere up to the ``studies`` folder holding a study's
    cells (in the workspace or the service root), not including it. Never at or above a
    workspace root or the service root, nor outside them.
    """
    current = folder.resolve()
    space = service.workspace()
    roots = [r.resolve() for r in space.roots]
    home = service.root.resolve()
    stop = _ui_user_folder(current, space.settings.workspace.ui_folder, roots)
    if stop is None:
        stop = next(
            (
                p
                for p in current.parents
                if p.name == "studies" and _inside(p, [*roots, home]) and p not in roots
            ),
            None,
        )
    if stop is None:
        return
    while stop in current.parents and current not in (*roots, home):
        try:
            if any(current.iterdir()):
                return
            # A writer may fill it, or another deletion remove it, between the two calls:
            # then it is not ours to remove, and the climb ends.
            current.rmdir()
        except OSError:
            return
        current = current.parent


def _ui_user_folder(path: Path, ui_folder: str | None, roots: Sequence[Path]) -> Path | None:
    """Return the web-UI user's folder ``path`` lies in, if any."""
    if not ui_folder or not roots:
        return None
    base = roots[0] / ui_folder
    if base not in path.parents:
        return None
    parts = path.relative_to(base).parts
    if parts and parts[0] == "owner":
        return base / "owner"
    if len(parts) >= 2 and parts[0] in ("guests", "agents"):
        return base / parts[0] / parts[1]
    return None
