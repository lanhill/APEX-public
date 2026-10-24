"""The application services: discovery, validation, identity, jobs, summaries.

Jobs run in real detached worker processes, on the generated linear-Gaussian cases.
"""

from collections.abc import Callable
from pathlib import Path
from typing import Protocol

import numpy as np
import pytest

from apex.algorithms.data_assimilation.gies import UNSCALED_DATA_ANOMALIES_WARNING
from apex.core.errors import ConfigError
from apex.services import composition
from apex.services.api import ApexService
from apex.services.identity import run_id
from apex.services.plugins import REGISTRIES, check_options
from apex.storage.run_directory import RunDirectory

TIMEOUT = 180.0
GIES = (
    'name = "gies"\nstrategy = { name = "rlmmac", options = { energy = 1.0 } }\n'
    "schedule = { data_mismatch_tolerance = 1e-9 }\nmax_iterations = 6"
)


class Case(Protocol):
    """What the ``linear_case`` fixture (conftest.py) returns."""

    directory: Path
    config: Path


CaseFactory = Callable[..., Case]


@pytest.fixture
def service(tmp_path: Path) -> ApexService:
    return ApexService(tmp_path / "service", max_concurrent_jobs=2)


def test_every_plugin_describes_and_checks_its_options() -> None:
    from apex.services.plugins import all_plugins

    infos = all_plugins()

    assert {(p.registry, p.name) for p in infos} == {
        (registry, name) for registry, entries in REGISTRIES.items() for name in entries.names()
    }
    for info in infos:
        # A misspelt option is refused for every plugin, before any run.
        assert check_options(info.registry, info.name, {"no_such_option": 1}), info.name
    gies = next(p for p in infos if p.name == "gies")
    assert gies.status == "parity"
    assert "max_trials_per_iteration" in gies.options["properties"]
    assert "lower_bounds" in gies.supplied
    assert check_options("simulators", "opm_flow", {"deck_template_file": "d", "timeout_s": "x"})
    # Core's methods, whatever other packages add (core tests pass with them).
    [unknown] = check_options("methods", "nope", {})
    assert unknown.startswith("unknown methods plugin 'nope'; available: ")
    available = set(unknown.split(": ", 1)[1].split(", "))
    assert {"chop", "es", "esmda", "gies", "gn_enrml", "lm_enrml", "multilevel_esmda"} <= available


def test_every_plugin_schema_is_json() -> None:
    """Option defaults must serialize; a dataclass default factory does not (it warned,
    and the schema showed no default; StandardModel.fluid_fits)."""
    import warnings

    from pydantic.json_schema import PydanticJsonSchemaWarning

    from apex.services.plugins import all_plugins

    with warnings.catch_warnings():
        warnings.simplefilter("error", PydanticJsonSchemaWarning)
        all_plugins()


def test_validation_plans_without_simulating(
    service: ApexService, linear_case: CaseFactory
) -> None:
    case = linear_case(ensemble_size=20)

    validation = service.validate(case.config)

    assert validation.valid, validation.errors
    plan = validation.plan
    assert plan is not None
    assert (plan.members, plan.n_state, plan.data) == (20, 3, {"history_matching": 8})
    assert plan.max_forecasts == 5  # ES-MDA: the prior and four steps
    assert plan.max_simulator_runs == 100
    assert not (case.directory / "runs").exists()  # nothing ran

    bad = service.validate(
        {"run": {"name": "x", "seed": 1}, "method": {"name": "esmda"}, "case": {"kind": "real"}}
    )
    assert not bad.valid
    assert any("state" in e for e in bad.errors)
    wrong = case.config.read_text().replace("energy = 1.0", "energy = 1.0, bogus = 2")
    (case.directory / "wrong.toml").write_text(wrong)
    assert service.validate(case.directory / "wrong.toml").errors == [
        "update_strategies 'approx': bogus: Extra inputs are not permitted"
    ]


@pytest.mark.parametrize(("scaled", "warned"), [("true", False), ("false", True)])
def test_unscaled_rlmmac_anomalies_are_warned_of_in_the_plan(
    service: ApexService, linear_case: CaseFactory, scaled: str, warned: bool
) -> None:
    # The user (2026-10-13): unscaled data anomalies do not follow the RLM-MAC theory.
    case = linear_case(
        ensemble_size=20,
        method='name = "gies"\nmax_iterations = 2\nstrategy = { name = "rlmmac", options = '
        f"{{ scale_data_anomalies = {scaled} }} }}",
    )

    plan = service.validate(case.config).plan

    assert plan is not None
    assert (UNSCALED_DATA_ANOMALIES_WARNING in plan.warnings) is warned


def test_run_ids_follow_content_not_names(linear_case: CaseFactory) -> None:
    case = linear_case()
    config = composition.load_config(case.config)
    renamed = config.model_copy(
        update={"run": config.run.model_copy(update={"name": "other", "output_dir": Path("/x")})}
    )
    reseeded = config.model_copy(update={"run": config.run.model_copy(update={"seed": 1})})

    assert run_id(renamed) == run_id(config)
    assert run_id(reseeded) != run_id(config)
    before = run_id(config)
    observations = case.directory / "observations.csv"
    observations.write_text(observations.read_text().replace("t0,y,", "t0,y,1"))
    assert run_id(composition.load_config(case.config)) != before  # the input's content


def test_a_run_job_completes_and_an_identical_one_is_reused(
    service: ApexService, linear_case: CaseFactory
) -> None:
    case = linear_case(ensemble_size=20)

    handle = service.submit_run(case.config, annotations={"hypothesis": "H0"})
    status = service.wait(handle.id, timeout_s=TIMEOUT, poll_s=0.2)

    assert status.state == "complete", service.events(handle.id)
    summary = service.run_summary(handle.run_ids[0])
    assert summary.state == "complete"
    assert summary.iterations == [0, 1, 2, 3, 4]
    assert summary.simulator_runs == 100
    assert summary.annotations["hypothesis"] == "H0"
    assert summary.provenance is not None
    assert summary.provenance["run_id"] == handle.run_ids[0]
    names = [e.event for e in service.events(handle.id)]
    assert names[:2] == ["job_queued", "job_started"]
    assert "forecast_done" in names
    assert names[-1] == "job_complete"

    again = service.submit_run(case.config)
    assert again.reused
    assert again.run_ids == handle.run_ids
    assert service.status(again.id).state == "complete"


def test_stop_at_a_checkpoint_then_resume_bit_identically(
    service: ApexService, linear_case: CaseFactory
) -> None:
    case = linear_case(method=GIES, ensemble_size=10)
    config = composition.load_config(case.config)
    assert config.run.output_dir is not None
    # A stop requested before the job starts: it stops at its first checkpoint.
    config.run.output_dir.mkdir(parents=True)
    (config.run.output_dir / "stop_requested").write_text("")

    handle = service.submit_run(config)
    stopped = service.wait(handle.id, timeout_s=TIMEOUT, poll_s=0.2)
    assert stopped.state == "stopped", service.events(handle.id)
    assert service.run_summary(config.run.output_dir).state == "stopped"

    resumed = service.wait(service.resume(handle.id).id, timeout_s=TIMEOUT, poll_s=0.2)
    assert resumed.state == "complete"

    # The same configuration elsewhere, uninterrupted.
    other = config.model_copy(
        update={"run": config.run.model_copy(update={"output_dir": case.directory / "other"})}
    )
    whole = service.submit_run(other, force_new=True)
    assert service.wait(whole.id, timeout_s=TIMEOUT, poll_s=0.2).state == "complete"
    # Every saved array and the whole history are identical to the uninterrupted run's.
    stopped_dir, whole_dir = (
        config.run.output_dir / "checkpoints",
        case.directory / "other" / "checkpoints",
    )
    names = sorted(
        p.name for p in whole_dir.glob("*.npz") if not p.name.startswith("algorithm_state")
    )
    assert names == sorted(
        p.name for p in stopped_dir.glob("*.npz") if not p.name.startswith("algorithm_state")
    )
    for name in names:
        with np.load(stopped_dir / name) as a, np.load(whole_dir / name) as b:
            assert a.files == b.files
            for key in a.files:
                np.testing.assert_array_equal(a[key], b[key], err_msg=f"{name}:{key}")
    assert (config.run.output_dir / "history.json").read_text() == (
        case.directory / "other" / "history.json"
    ).read_text()


def test_a_client_configuration_without_a_place_goes_under_the_root(
    service: ApexService, linear_case: CaseFactory
) -> None:
    import tomllib

    case = linear_case(ensemble_size=10)
    document = tomllib.loads(case.config.read_text())

    handle = service.submit_run(document, base_dir=case.directory)
    status = service.wait(handle.id, timeout_s=TIMEOUT, poll_s=0.2)

    assert status.state == "complete"
    assert Path(status.run_dirs[0]).parent == service.root / "runs"
    changed = dict(document)
    changed["run"] = {**document["run"], "seed": 99, "output_dir": status.run_dirs[0]}
    with pytest.raises(ConfigError, match="already holds another run"):
        service.submit_run(changed, base_dir=case.directory)


def test_queued_jobs_can_be_cancelled_and_failures_are_recorded(
    tmp_path: Path, linear_case: CaseFactory
) -> None:
    service = ApexService(tmp_path / "one", max_concurrent_jobs=1)
    first = service.submit_run(linear_case(name="a", ensemble_size=10).config)
    # Another run (an identical one would be the queued first job, reused).
    other = (
        'name = "esmda"\noptions = { inflation = [2, 4, 8, 8] }\n'
        'strategy = { name = "approx", options = { energy = 1.0 } }'
    )
    queued = service.submit_run(linear_case(name="b", ensemble_size=10, method=other).config)
    service.stop(queued.id)
    assert service.status(queued.id).state in ("cancelled", "running", "complete")
    assert service.wait(first.id, timeout_s=TIMEOUT, poll_s=0.2).state == "complete"

    broken = linear_case(name="c", ensemble_size=10, simulator_extra="fail_members = [0, 1, 2]")
    failed = service.wait(service.submit_run(broken.config).id, timeout_s=TIMEOUT, poll_s=0.2)
    assert failed.state == "failed"
    assert failed.error is not None
    assert "members simulated successfully" in failed.error
    # Recorded in the root's store, and not retried: the failure would repeat.
    record = service.jobs.store.job(failed.id)
    assert record["state"] == "failed"
    assert record["retries"] == 0


@pytest.mark.parametrize("retries", [None, 3])
def test_the_plan_bounds_the_simulator_runs_of_gies(
    service: ApexService, linear_case: CaseFactory, retries: int | None
) -> None:
    """GIES with RLM-MAC (mean-model forecasts), with and without inner retries."""
    options = "" if retries is None else f"options = {{ max_trials_per_iteration = {retries} }}\n"
    case = linear_case(
        method=GIES.replace("max_iterations = 6", "max_iterations = 4") + "\n" + options,
        ensemble_size=10,
        simulator_extra="fail_members = [2]",  # a failure: reduced-member mean forecasts too
        failures="[failures]\nmin_success_fraction = 0.8",
    )
    plan = service.validate(case.config).plan
    assert plan is not None
    assert plan.max_simulator_runs is not None

    handle = service.submit_run(case.config)
    assert service.wait(handle.id, timeout_s=TIMEOUT, poll_s=0.2).state == "complete"

    used = service.run_summary(handle.run_ids[0]).simulator_runs
    assert 0 < used <= plan.max_simulator_runs


def test_the_run_id_covers_the_model_directory_and_the_code(
    linear_case: CaseFactory, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import apex.services.identity as identity

    case = linear_case()
    config = composition.load_config(case.config)
    template = tmp_path / "model" / "deck.mako"
    template.parent.mkdir()
    template.write_text("INCLUDE '${case_dir}/grid.inc' /")
    (template.parent / "grid.inc").write_text("DX 1 /")
    (template.parent / "runs").mkdir()
    with_deck = config.model_copy(
        update={
            "simulator": config.simulator.model_copy(
                update={
                    "options": {**config.simulator.options, "deck_template_file": str(template)}
                }
            )
        }
    )
    first = run_id(with_deck)
    (template.parent / "runs" / "output.txt").write_text("results are not inputs")
    assert run_id(with_deck) == first
    (template.parent / "grid.inc").write_text("DX 2 /")  # an INCLUDEd file changed
    assert run_id(with_deck) != first

    before = run_id(config)
    monkeypatch.setattr(identity, "code_version", lambda: {"apex": "0.2", "commit": "x"})
    assert run_id(config) != before  # other code: no reuse of old results


def test_a_run_that_failed_before_any_checkpoint_runs_again_rather_than_resume(
    tmp_path: Path, linear_case: CaseFactory
) -> None:
    """A client's report (2026-09-29): an identical resubmission of a run that failed in its first
    forecast was "reused" as a resume, which failed at once ("no saved state to resume
    from"). With nothing to resume, the run starts again in its own folder."""
    service = ApexService(tmp_path / "root")
    broken = linear_case(name="early", ensemble_size=10, simulator_extra="fail_members = [0, 1, 2]")
    submitted = service.submit_run(broken.config)
    first = service.wait(submitted.id, timeout_s=TIMEOUT, poll_s=0.2)
    assert first.state == "failed"
    run_dir = broken.directory / "runs" / "early"
    assert not (run_dir / "state.json").exists()  # it failed before its first checkpoint

    again = service.submit_run(broken.config)

    assert (again.kind, again.reused) == ("run", False)
    assert again.run_ids == submitted.run_ids  # the same run, in its own folder
    status = service.wait(again.id, timeout_s=TIMEOUT, poll_s=0.2)
    assert status.state == "failed"  # the same members fail again: a real rerun
    assert status.error is not None
    assert "no saved state" not in status.error
    assert "members simulated successfully" in status.error


def test_a_stop_asked_while_a_resume_starts_is_kept(linear_case: CaseFactory) -> None:
    """The review of the follow-ups: a resume withdraws the run's stop file once it
    holds the lock; a job's stop asked for before that (while the run was built) is
    requested again, so the run stops at its next checkpoint instead of running on."""
    case = linear_case(ensemble_size=20)
    composition.start(case.config, max_steps=1)
    run_dir = case.directory / "runs" / "case"
    (run_dir / "stop_requested").write_text("")  # written while the resume was built

    result = composition.resume(run_dir, stopped=lambda: True)

    assert not result.complete
    # The prior, the first step, and the one the stop let finish (of four).
    assert [r["iteration"] for r in RunDirectory(run_dir).history_dicts()] == [0, 1, 2]


def test_a_run_without_a_checkpoint_says_so_when_resumed(
    service: ApexService, linear_case: CaseFactory
) -> None:
    """The review of the follow-ups: a run that ended before its first checkpoint
    (stopped before its worker started, or failed) has nothing to continue; both the
    service and composition say so, instead of failing later in the worker."""
    from apex.core.errors import WorkflowError

    case = linear_case(name="early", ensemble_size=10, simulator_extra="fail_members = [0, 1, 2]")
    submitted = service.submit_run(case.config)
    assert service.wait(submitted.id, timeout_s=TIMEOUT, poll_s=0.2).state == "failed"
    run_dir = case.directory / "runs" / "early"

    with pytest.raises(ConfigError, match="no checkpoint to resume from"):
        service.resume(submitted.id)
    with pytest.raises(WorkflowError, match="no checkpoint to resume from"):
        composition.resume(run_dir)


def test_a_stop_asked_while_a_run_starts_is_kept(linear_case: CaseFactory) -> None:
    """The coder's review: as for a resume, a new run whose job was asked to stop before it
    held its lock requests the stop once it does, and stops at its first checkpoint (the
    prior's)."""
    case = linear_case(ensemble_size=20)

    result = composition.start_config(composition.load_config(case.config), stopped=lambda: True)

    assert not result.complete
    run_dir = RunDirectory(case.directory / "runs" / "case")
    assert [r["iteration"] for r in run_dir.history_dicts()] == [0]


def test_an_executor_not_installed_has_none_of_its_options_hashed(
    linear_case: CaseFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The coder's review: with an executor's package missing (a stale
    environment, or an install without a workspace member opening a configuration naming its
    executor), which of its options are locations is unknown; none of them is read from disk
    (a case-studies folder of tens of GB was hashed for minutes), and the id stays defined."""
    from apex.domain.config import PluginSection
    from apex.services import identity

    case = linear_case(ensemble_size=10)
    folder = case.directory.parent
    execution = PluginSection(name="not_installed", options={"local_case_dir": str(folder)})
    config = composition.load_config(case.config).model_copy(update={"execution": execution})

    def tree_hash(directory: Path) -> str:
        raise AssertionError(f"{directory} was walked")

    monkeypatch.setattr(identity, "tree_hash", tree_hash)
    first = identity.run_id(config)
    (folder / "another_study.txt").write_text("unrelated\n")
    assert identity.run_id(config) == first
