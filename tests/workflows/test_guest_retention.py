"""The guests' retention and storage limits.

A guest submits runs and a study through the REST API; the owner has runs of their own.
Pruning must delete only the guest's finished work (directories, catalogue entries, study
records, jobs and their events), by age and, beyond the storage limit, oldest first;
never the owner's work, a run the guest merely reused, or queued and running jobs.
"""

import datetime as dt
from collections.abc import Callable
from pathlib import Path
from typing import Literal, Protocol

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from apex.api.app import create_app
from apex.services.api import ApexService
from apex.services.auth import ui_token
from apex.services.daemon import prune_guest_work
from apex.services.examples import Example, get_example
from apex.services.guest_retention import prune_guests, submissions
from apex.services.settings import GuestsSection

TIMEOUT = 180.0
ACCESS = """
[workspace]
folders = ["alpha"]
[plugins]
methods = ["*"]
update_strategies = ["*"]
localizations = ["*"]
simulators = ["*"]
executors = ["*"]
priors = ["*"]
metrics = ["*"]
[actions]
submit = true
"""


class Case(Protocol):
    """What the ``linear_case`` fixture (conftest.py) returns."""

    directory: Path
    config: Path


CaseFactory = Callable[..., Case]


@pytest.fixture
def world(tmp_path: Path, linear_case: CaseFactory) -> tuple[ApexService, dict[str, str]]:
    root = tmp_path / "service"
    root.mkdir()
    (root / "service.toml").write_text(f'[workspace]\nroots = ["{tmp_path}"]\n')
    (root / "ui_access.toml").write_text(ACCESS)
    service = ApexService(root)
    owner = service.submit_run(linear_case(name="alpha", ensemble_size=10).config, client="ui")
    service.wait(owner.id, timeout_s=TIMEOUT, poll_s=0.2)
    http = TestClient(create_app(root))
    guest = {"Authorization": f"Bearer {ui_token(root, create=True, guest='alice')}"}
    text = (tmp_path / "alpha" / "config.toml").read_text()
    ids = {"owner": owner.id, "owner_run": owner.run_ids[0]}
    for name, members in (("old", 11), ("new", 12)):
        path = tmp_path / "alpha" / f"{name}.toml"
        path.write_text(
            text.replace('name = "alpha"', f'name = "{name}"').replace(
                "ensemble_size = 10", f"ensemble_size = {members}"
            )
        )
        handle = http.post("/v1/runs", headers=guest, json={"config_file": str(path)}).json()
        assert "id" in handle, handle
        service.wait(handle["id"], timeout_s=TIMEOUT, poll_s=0.2)
        ids[name], ids[f"{name}_run"] = handle["id"], handle["run_ids"][0]
    (tmp_path / "alpha" / "study.toml").write_text(
        '[study]\nname = "s"\n[study.cases]\nalpha = "old.toml"\n'
        '[[study.variants]]\nname = "baseline"\n[[study.variants]]\nname = "loc"\n'
        'patch = { method = { localization = { name = "adaptive" } } }\n'
    )
    study_file = str(tmp_path / "alpha" / "study.toml")
    study = http.post("/v1/studies", headers=guest, json={"study_file": study_file}).json()
    assert "id" in study, study
    service.wait(study["id"], timeout_s=TIMEOUT, poll_s=0.2)
    ids["study"] = study["id"]
    # A guest's job that reused the owner's run, as submissions did (a guest's
    # identical submission now runs again in their own folder).
    owner_dir = service.store.find(owner.run_ids[0]).path  # type: ignore[union-attr]
    ids["reuse"] = service.jobs.submit(
        "run", {"reused": str(owner_dir)}, run_ids=owner.run_ids, run_dirs=[str(owner_dir)],
        annotations={"submitted_by": "guest:alice"}, state="complete", client="ui",
    )  # fmt: skip
    return service, ids


def _age(service: ApexService, job: str, days: float) -> None:
    record = service.jobs.record(job)
    stamp = (dt.datetime.now(dt.UTC) - dt.timedelta(days=days)).isoformat(timespec="seconds")
    service.jobs.update(job, created=stamp)
    assert record["id"] == job


def test_the_inventory_holds_only_what_the_guest_created(
    world: tuple[ApexService, dict[str, str]],
) -> None:
    service, ids = world

    units = {u.job: u for u in submissions(service)}

    assert set(units) == {ids["old"], ids["new"], ids["study"], ids["reuse"]}
    assert units[ids["reuse"]].run_ids == []  # the owner's run, only reused
    assert len(units[ids["study"]].jobs) == 3  # the study and its two cells
    assert len(units[ids["study"]].dirs) == 2
    assert all(u.size > 0 for u in units.values() if u.dirs)


def test_after_thirty_days_a_guests_work_is_deleted_and_nobody_elses(
    world: tuple[ApexService, dict[str, str]],
) -> None:
    service, ids = world
    old_dir = service.store.find(ids["old_run"]).path  # type: ignore[union-attr]
    study_dirs = next(u for u in submissions(service) if u.job == ids["study"]).dirs
    for job in (ids["old"], ids["study"], ids["reuse"]):
        _age(service, job, 31)
    _age(service, ids["owner"], 400)  # the owner's: never pruned

    deleted = prune_guests(service)

    # The study's baseline cell reused the guest's own "old" run: one unit, deleted together.
    assert {j for u in deleted for j in u.jobs} >= {ids["old"], ids["study"], ids["reuse"]}
    assert ids["new"] not in {j for u in deleted for j in u.jobs}
    assert not old_dir.exists()
    assert not any(d.exists() for d in study_dirs)
    assert service.store.find(ids["old_run"]) is None
    assert service.jobs.store.study("s") is None
    assert not (service.root / "studies" / "s").exists()  # study.json went too
    remaining = {r["id"] for r in service.jobs.all_records()}
    assert ids["old"] not in remaining
    assert service.jobs.store.events(ids["old"]) == []
    # Kept: the guest's newer run, the owner's run (reused by the guest) and job.
    assert {ids["new"], ids["owner"]} <= remaining
    assert service.store.find(ids["new_run"]) is not None
    assert service.store.find(ids["owner_run"]) is not None


def test_over_the_storage_limit_the_oldest_goes_first(
    world: tuple[ApexService, dict[str, str]],
) -> None:
    service, ids = world
    units = {u.job: u for u in submissions(service)}
    # Submitted in order: old, new, the study, the reuse (no files of its own). The study
    # reused "old" (its baseline cell), so the two are one unit, as recent as the study and
    # counted once. A limit that unit fits in alone: "new", now the oldest, must go.
    rules = GuestsSection(max_storage_mb=units[ids["study"]].size / 1e6, retention_days=30)

    deleted = prune_guests(service, rules)

    assert [u.job for u in deleted] == [ids["new"]]
    assert service.store.find(ids["new_run"]) is None
    assert service.store.find(ids["old_run"]) is not None
    assert {r["id"] for r in service.jobs.all_records()} >= {ids["old"], ids["study"], ids["reuse"]}
    assert service.store.find(ids["owner_run"]) is not None


def test_queued_and_running_work_is_never_deleted(
    world: tuple[ApexService, dict[str, str]],
) -> None:
    service, ids = world
    service.jobs.update(ids["old"], state="running")
    _age(service, ids["old"], 60)

    deleted = prune_guests(service, GuestsSection(max_storage_mb=0.000001, retention_days=30))

    assert ids["old"] not in {u.job for u in deleted}
    assert ids["old"] in {r["id"] for r in service.jobs.all_records()}


def test_the_service_applies_the_limits(world: tuple[ApexService, dict[str, str]]) -> None:
    service, ids = world
    _age(service, ids["new"], 45)
    (service.root / "service.toml").write_text(
        (service.root / "service.toml").read_text() + "[guests]\nretention_days = 40\n"
    )

    assert prune_guest_work(service.root) == 1  # [guests] retention_days = 40
    assert service.store.find(ids["new_run"]) is None


def _optimization(service: ApexService, name: str, size: int) -> Path:
    folder = service.root / "optimizations" / name / "abc"
    (folder / "checkpoints").mkdir(parents=True)
    (folder / "checkpoints" / "iter_0.npz").write_bytes(b"x" * size)
    return folder


def _job(
    service: ApexService,
    kind: str,
    folder: Path,
    state: Literal["queued", "running", "complete"],
    owner: str,
) -> str:
    return service.jobs.submit(
        kind, {}, run_ids=[folder.name], run_dirs=[str(folder)], state=state, client="ui",
        annotations={"submitted_by": owner} if owner else {},
    )  # fmt: skip


def test_guests_optimizations_are_pruned_unless_being_resumed(
    world: tuple[ApexService, dict[str, str]],
) -> None:
    """Optimization outputs are run_dirs, not catalogued runs. A resumed
    optimization's job shares the directory: the two are one unit, active while either
    is, as old as the newer, and deleted together.
    """
    service, _ = world
    folder = _optimization(service, "opt", 1000)
    first = _job(service, "optimization", folder, "complete", "guest:alice")
    resumed = _job(service, "optimization_resume", folder, "running", "guest:alice")
    units = {u.job: u for u in submissions(service)}
    assert units[first].dirs == [folder]
    assert units[first].size >= 1000
    _age(service, first, 31)
    _age(service, resumed, 31)

    assert not {first, resumed} & {u.job for u in prune_guests(service)}  # resume running
    assert folder.exists()
    service.jobs.update(resumed, state="complete")
    _age(service, resumed, 5)  # finished recently: the unit is as old as its newest job
    assert not {first, resumed} & {u.job for u in prune_guests(service)}
    assert folder.exists()
    _age(service, resumed, 31)
    prune_guests(service)
    remaining = {r["id"] for r in service.jobs.all_records()}
    assert not {first, resumed} & remaining  # deleted together
    assert not folder.exists()


def test_a_shared_optimization_directory_counts_once_against_the_limit(
    world: tuple[ApexService, dict[str, str]],
) -> None:
    service, _ = world
    folder = _optimization(service, "opt", 400_000)
    first = _job(service, "optimization", folder, "complete", "guest:alice")
    again = _job(service, "optimization", folder, "complete", "guest:alice")  # same config
    units = {u.job: u for u in submissions(service)}
    others = sum(u.size for u in units.values() if u.job not in (first, again))
    # Room for everything if the directory counts once, not if it counts twice.
    limit = (others + units[first].size + 1000) / 1e6
    deleted = prune_guests(service, GuestsSection(max_storage_mb=limit, retention_days=30))
    assert deleted == []
    assert folder.exists()


def test_owners_and_local_optimizations_are_never_pruned(
    world: tuple[ApexService, dict[str, str]],
) -> None:
    service, _ = world
    folder = _optimization(service, "mine", 1000)
    job = _job(service, "optimization", folder, "complete", "")
    _age(service, job, 400)
    prune_guests(service, GuestsSection(max_storage_mb=0.000001, retention_days=1))
    assert folder.exists()
    assert job in {r["id"] for r in service.jobs.all_records()}


def test_a_symlinked_folder_loses_its_link_and_a_failed_removal_keeps_the_records(
    world: tuple[ApexService, dict[str, str]], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Nothing is left on disk without a record that leads to it (review of 2026-10-04)."""
    import shutil

    service, _ = world
    target = service.root / "elsewhere"
    (target / "checkpoints").mkdir(parents=True)
    link = service.root / "optimizations" / "linked" / "abc"
    link.parent.mkdir(parents=True)
    link.symlink_to(target, target_is_directory=True)
    linked = _job(service, "optimization", link, "complete", "guest:alice")
    _age(service, linked, 31)
    assert linked in {u.job for u in prune_guests(service)}
    assert not link.is_symlink()
    assert target.is_dir()  # the files it pointed to are not the guest's to delete

    folder = _optimization(service, "stuck", 1000)
    stuck = _job(service, "optimization", folder, "complete", "guest:alice")
    _age(service, stuck, 31)
    monkeypatch.setattr(shutil, "rmtree", lambda *args, **kwargs: None)  # removal fails
    assert stuck not in {u.job for u in prune_guests(service)}
    assert folder.exists()
    assert stuck in {r["id"] for r in service.jobs.all_records()}  # its record stays


def test_expired_work_leaves_no_empty_folders_but_the_guests_own(
    world: tuple[ApexService, dict[str, str]],
) -> None:
    """Review of 2026-10-05: pruning removes emptied runs/ and studies/ folders, not the
    guest's own folder."""
    service, ids = world
    for name in ("old", "new", "study", "reuse"):
        _age(service, ids[name], 31)
    alice = service.workspace().roots[0] / "UI" / "guests" / "alice"
    assert (alice / "runs").is_dir()

    prune_guests(service)

    assert not (alice / "runs").exists()
    assert not (alice / "studies").exists()
    assert alice.is_dir()


def test_what_could_not_be_deleted_still_counts_against_the_limit(
    world: tuple[ApexService, dict[str, str]], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Review of 2026-10-05: an expired submission whose removal fails still takes the
    guest's space, and a submission counts as freed only by what was really removed."""
    import shutil

    service, _ = world
    stuck_folder = _optimization(service, "stuck", 1000)
    stuck = _job(service, "optimization", stuck_folder, "complete", "guest:carol")
    _age(service, stuck, 31)
    newer_folder = _optimization(service, "newer", 600)
    newer = _job(service, "optimization", newer_folder, "complete", "guest:carol")
    real = shutil.rmtree

    def rmtree(path: Path, *args: object, **kwargs: object) -> None:
        if Path(path) != stuck_folder:
            real(path, ignore_errors=True)

    monkeypatch.setattr(shutil, "rmtree", rmtree)
    rules = GuestsSection(max_storage_mb=1200 / 1e6, retention_days=30)

    deleted = {u.job for u in prune_guests(service, rules)}

    # 1000 kept + 600 = 1600 > 1200: the newer one goes; before, the kept 1000 bytes were
    # not counted and nothing was deleted.
    assert stuck not in deleted
    assert stuck_folder.exists()
    assert newer in deleted
    assert not newer_folder.exists()


def test_work_kept_because_it_is_in_use_does_not_push_newer_work_out(
    world: tuple[ApexService, dict[str, str]], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Review of 2026-10-05: an expired submission kept only because a process holds it (a
    prediction on an old run) is not counted, so the guest's newer work stays."""
    from apex.services import guest_retention

    service, _ = world
    held_folder = _optimization(service, "held", 1000)
    held = _job(service, "optimization", held_folder, "complete", "guest:dora")
    _age(service, held, 31)
    newer_folder = _optimization(service, "newer", 600)
    newer = _job(service, "optimization", newer_folder, "complete", "guest:dora")
    monkeypatch.setattr(guest_retention, "locked", lambda path: Path(path) == held_folder)
    rules = GuestsSection(max_storage_mb=1200 / 1e6, retention_days=30)

    deleted = {u.job for u in prune_guests(service, rules)}

    assert held not in deleted
    assert newer not in deleted
    assert newer_folder.exists()


# ------------------------------------------------- copies of the examples (revision 2)


def _copies_world(tmp_path: Path) -> tuple[ApexService, Path, Example]:

    root = tmp_path / "service"
    root.mkdir()
    (root / "service.toml").write_text(f'[workspace]\nroots = ["{tmp_path}"]\n')
    service = ApexService(root)
    folder = service.workspace().ui_user_folder("alice")
    assert folder is not None
    examples = Path(__file__).resolve().parents[2] / "examples"
    return service, folder / "examples", get_example(examples, "toy/rastrigin")


def _stamp(copy: Path, days: float) -> None:
    import json

    from apex.services.examples import COPY_MARKER

    created = dt.datetime.now(dt.UTC) - dt.timedelta(days=days)
    marker = copy / COPY_MARKER
    record = json.loads(marker.read_text())
    marker.write_text(json.dumps({**record, "created": created.isoformat()}))


def test_an_expired_example_copy_is_deleted_and_a_recent_one_kept(tmp_path: Path) -> None:
    """A guest's copy goes after ``retention_days``, dated by its marker (revision 2)."""
    from apex.services.examples import COPY_MARKER, copy_example

    service, examples, example = _copies_world(tmp_path)
    old = copy_example(example, examples).folder
    new = copy_example(example, examples).folder
    assert (old / COPY_MARKER).is_file()
    _stamp(old, 40)
    deleted = prune_guests(service, GuestsSection(retention_days=30, max_storage_mb=500))
    assert [u.job for u in deleted] == [f"example:examples/{old.relative_to(examples)}"]
    assert not old.exists()
    assert new.is_dir()


def test_example_copies_count_against_the_storage_limit(tmp_path: Path) -> None:
    from apex.services.examples import copy_example

    service, examples, example = _copies_world(tmp_path)
    first = copy_example(example, examples).folder
    second = copy_example(example, examples).folder
    (first / "padding.bin").write_bytes(b"\0" * 600_000)
    (second / "padding.bin").write_bytes(b"\0" * 600_000)
    _stamp(first, 2)
    _stamp(second, 1)
    prune_guests(service, GuestsSection(retention_days=30, max_storage_mb=1))
    assert not first.exists()  # oldest first, down to the limit
    assert second.is_dir()


def test_a_copy_holding_a_catalogued_run_is_kept_and_not_charged_for_it(tmp_path: Path) -> None:
    """A run started from a copy writes inside it: the copy waits for the run to go."""
    from apex.core.storage import CatalogueEntry
    from apex.services.examples import copy_example
    from apex.services.guest_retention import example_copies

    service, examples, example = _copies_world(tmp_path)
    copy = copy_example(example, examples).folder
    run = copy / "runs" / "r"
    run.mkdir(parents=True)
    (run / "big.bin").write_bytes(b"\0" * 2_000_000)
    service.store.register(CatalogueEntry("f" * 64, run, "r", "2026-01-01T00:00:00+00:00", {}))
    _stamp(copy, 40)
    [unit] = example_copies(service, "guest:", [])
    assert unit.active
    assert unit.size < 1_000_000
    assert prune_guests(service, GuestsSection(retention_days=30, max_storage_mb=500)) == []
    assert run.is_dir()


def test_a_copy_a_run_reads_from_elsewhere_is_kept(tmp_path: Path) -> None:
    """The web UI places a run in the user's runs/ folder, not in the copy, yet the run reads
    the copy's files (its case_dir): the copy stays while a record names it (review)."""
    from apex.core.storage import CatalogueEntry
    from apex.services.examples import copy_example

    service, examples, example = _copies_world(tmp_path)
    copy = copy_example(example, examples).folder
    run = examples.parent / "runs" / "r"
    run.mkdir(parents=True)
    (run / "config.json").write_text(f'{{"simulator": {{"case_dir": "{copy}"}}}}')
    service.store.register(CatalogueEntry("e" * 64, run, "r", "2026-01-01T00:00:00+00:00", {}))
    _stamp(copy, 40)

    assert prune_guests(service, GuestsSection(retention_days=30, max_storage_mb=0.001)) == []
    assert copy.is_dir()


def test_a_copy_a_queued_job_names_only_inside_its_spec_is_kept(tmp_path: Path) -> None:
    """A queued job's spec holds its configuration as JSON text, so the copy's path ends in
    an escaped quote there (second review: it was missed, and the copy deleted under it)."""
    import json

    from apex.services.examples import copy_example
    from apex.services.guest_retention import example_copies

    service, examples, example = _copies_world(tmp_path)
    copy = copy_example(example, examples).folder
    _stamp(copy, 40)
    config = json.dumps({"simulator": {"options": {"case_dir": str(copy.resolve())}}})
    record = {"id": "j", "state": "queued", "spec": {"config": config}}
    service.jobs.all_records = lambda: [record]  # type: ignore[method-assign]

    [unit] = example_copies(service, "guest:", [])
    assert unit.active
    record["state"] = "complete"  # finished: its run's catalogue entry would name it
    [unit] = example_copies(service, "guest:", [])
    assert not unit.active
