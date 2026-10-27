"""The application service every front end calls.

:class:`ApexService` takes and returns the typed documents of
:mod:`apex.domain.services`, so the CLI, the Python client (:mod:`apex.client`) and, later,
the REST API and MCP server are translations of the same calls.

Its root (``$APEX_HOME``, else ``~/.apex/service``) holds the run catalogue, the jobs,
and the runs of studies and of configurations submitted without an output directory.
"""

import json
import logging
import os
import typing
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from pydantic import BaseModel, ValidationError

from apex.core.engines import EngineMethod, RunEngine
from apex.core.errors import ApexError, ConfigError, validation_problems
from apex.core.plugins import engines, simulators
from apex.core.storage import CatalogueEntry, RunStore
from apex.domain.config import HistoryMatchConfig, PluginSection
from apex.domain.optimization import OptimizationConfig
from apex.domain.prediction import PERCENTILE_LABELS, PredictionConfig
from apex.domain.services import (
    Annotations,
    Event,
    JobHandle,
    JobStatus,
    PluginInfo,
    PredictionSummary,
    RunSteps,
    RunSummary,
    StepInfo,
    StepStatistics,
    Validation,
)
from apex.services import composition, policy, prediction

if TYPE_CHECKING:
    from apex.domain import summary as step_summary
    from apex.domain.optimization_study import OptimizationStudySpec
    from apex.domain.services import Comparison, OptimizationSummary
    from apex.domain.study import StudySpec
    from apex.services.study import ApprovalHook
    from apex.services.workspace import Workspace
from apex.services.access import SUBMITTED_BY
from apex.services.approval import ApprovalRequired, Approvals, Grants, fingerprint
from apex.services.code import CodeCopies, CodeError, run_commit
from apex.services.identity import SHORT, owned_run_id, run_id
from apex.services.jobs import FINAL, LocalJobRunner, service_pid
from apex.services.plugins import all_plugins
from apex.services.validation import case_key, validate_history_match
from apex.storage import optimization_directory as opt_layout
from apex.storage.catalogue import now
from apex.storage.locks import in_use
from apex.storage.run_directory import RunDirectory
from apex.storage.run_results import RunResults
from apex.storage.sqlite_store import JOB_PROTOCOL

logger = logging.getLogger(__name__)


def default_root() -> Path:
    """Return the services root: ``$APEX_HOME``, else ``~/.apex/service``.

    An empty ``APEX_HOME`` counts as unset: ``Path("")`` is the working directory, which
    would put a stray ``apex.db`` wherever a command ran.
    """
    return Path(os.environ.get("APEX_HOME") or Path.home() / ".apex" / "service")


#: The configuration kinds whose JSON Schema the API, the tools and the clients give.
type SchemaKind = Literal[
    "history_match", "prediction", "study", "optimization", "optimization_study"
]
#: ``apex schema`` also gives ``service`` (``<root>/service.toml``): the command line only,
#: since the service's schema names every installed agent's settings, private ones too,
#: which web-UI guests must not see.
SCHEMA_KINDS: tuple[str, ...] = (*typing.get_args(SchemaKind.__value__), "service")


def config_schema(kind: str) -> dict[str, Any]:
    """Return the JSON Schema of a configuration kind (:data:`SCHEMA_KINDS`).

    Needs no service: no root, no ``service.toml`` read.

    Raises
    ------
    ConfigError
        If ``kind`` is not one of :data:`SCHEMA_KINDS`.
    """
    from apex.domain.optimization import OptimizationConfig
    from apex.domain.optimization_study import OptimizationStudySpec
    from apex.domain.study import StudySpec
    from apex.services.settings import ServiceSettings

    models: dict[str, type[BaseModel]] = {
        "history_match": HistoryMatchConfig,
        "prediction": PredictionConfig,
        "study": StudySpec,
        "optimization": OptimizationConfig,
        "optimization_study": OptimizationStudySpec,
        "service": ServiceSettings,
    }
    if kind not in models:
        raise ConfigError(f"unknown schema {kind!r}; use one of {', '.join(SCHEMA_KINDS)}")
    schema: dict[str, Any] = models[kind].model_json_schema()
    return schema


#: Why a run is refused when its output folder holds another run (submit and validation).
OUTPUT_TAKEN = "{output} already holds another run; give this one another name or output_dir"


class ApexService:
    """APEX's use cases: discover, validate, submit, follow, stop, resume, summarise.

    Parameters
    ----------
    root
        The services root (default :func:`default_root`).
    max_concurrent_jobs
        Jobs running at once (default: ``service.toml``, else 2).
    approval_threshold_runs
        Overrides ``[approval] max_simulator_runs`` of ``service.toml``.
    store
        The run catalogue (default: ``<root>/catalogue.jsonl``).
    """

    def __init__(
        self,
        root: Path | None = None,
        *,
        max_concurrent_jobs: int | None = None,
        store: RunStore | None = None,
        approval_threshold_runs: int | None = None,
    ) -> None:
        self.root = Path(root) if root is not None else default_root()
        self.jobs = LocalJobRunner(self.root, max_concurrent_jobs)
        self.store: RunStore = store if store is not None else self.jobs.store
        settings = self.jobs.settings.approval
        if approval_threshold_runs is not None:
            settings = settings.model_copy(update={"max_simulator_runs": approval_threshold_runs})
        self.approvals = Approvals(self.jobs.store, settings)
        self.grants = Grants(self.jobs.store)
        self.code = CodeCopies.from_settings(self.jobs.settings.jobs, self.root)

    # ------------------------------------------------------------------ capabilities

    def pin_code(self, code: str | None, run_dir: Path | None = None) -> dict[str, str] | None:
        """Return the code pin of a submission, freezing its copy first.

        ``code`` is ``None`` or ``"live"`` (no pin: the checkout as it is when the worker
        starts), ``"frozen"`` (the commit ``run_dir``'s run started on, when it has one and
        it was clean, else the repository's HEAD), or a commit.

        Raises
        ------
        CodeError
            If the commit cannot be pinned or its copy made.
        """
        if code is None or code == "live":
            return None
        self._check_service_takes_pins()
        if code != "frozen":
            return self.code.ensure(code).as_json()
        started = None if run_dir is None else run_commit(run_dir)
        if started is not None:
            # A resumed run continues on its own code, so that it stays bit-identical.
            try:
                return self.code.ensure(started).as_json()
            except CodeError as error:
                logger.warning(
                    "%s started on %s, which cannot be pinned (%s): it continues on HEAD",
                    run_dir, started, error, extra={"event": "code_pin_fallback"},
                )  # fmt: skip
        return self.code.ensure("HEAD").as_json()

    def study_code(self, code: str | None) -> str | None:
        """Return the one commit every cell of a study runs (``None``: live code).

        ``"frozen"`` is the repository's HEAD now, frozen once, so that no cell runs other
        code than the others (not each reused run's own commit, as :meth:`pin_code`).
        """
        if code is None or code == "live":
            return None
        self._check_service_takes_pins()
        return self.code.ensure("HEAD" if code == "frozen" else code).commit

    def _check_service_takes_pins(self) -> None:
        """Refuse a pin the running service would ignore.

        A service of code from before pinned jobs (or of another job protocol) starts every
        worker with its own interpreter.

        Raises
        ------
        CodeError
            If the root's service runs and has not declared this job protocol.
        """
        pid = service_pid(self.root)
        if pid is None or self.jobs.store.service_protocol(pid) == JOB_PROTOCOL:
            return
        raise CodeError(
            f"the running service (pid {pid}) predates frozen jobs or speaks another job "
            "protocol: it would start this job on its own code. Restart it (apex service "
            "stop, then start) or submit with --code live"
        )

    def plugins(self) -> list[PluginInfo]:
        """Return every registered plugin, with its status and option schema."""
        return all_plugins()

    def schema(self, kind: SchemaKind) -> dict[str, Any]:
        """Return the JSON Schema of a configuration kind (:func:`config_schema`).

        Raises
        ------
        ConfigError
            For any other kind: ``service`` only the command line gives (:data:`SCHEMA_KINDS`).
        """
        kinds = typing.get_args(SchemaKind.__value__)
        if kind not in kinds:
            raise ConfigError(f"unknown schema {kind!r}; use one of {', '.join(kinds)}")
        return config_schema(kind)

    # ------------------------------------------------------------------ before spending compute

    def resolve(
        self, config: HistoryMatchConfig | Mapping[str, Any] | Path, base_dir: Path | None = None
    ) -> HistoryMatchConfig:
        """Return a history-matching configuration with its paths resolved.

        Raises
        ------
        ConfigError
            If it is not a valid configuration.
        """
        rules = self.jobs.settings.policy
        if isinstance(config, HistoryMatchConfig):
            policy.check(self._placed(config), rules, self.root)
            return config
        if isinstance(config, Path):
            policy.check_file(config, rules, self.root)  # before it is opened
            loaded = composition.load_config(config)
            policy.check(self._placed(loaded), rules, self.root)
            return loaded
        try:
            model = HistoryMatchConfig.model_validate(dict(config))
        except ValidationError as error:
            raise ConfigError(f"invalid configuration:\n{validation_problems(error)}") from error
        base = (base_dir or Path.cwd()).resolve()
        # First the policy: computing the run's identity below builds its
        # simulator and runs its executable, so nothing may happen before this check.
        policy.check(composition._resolve_paths(self._placed(model), base), rules, self.root)
        if model.run.output_dir is None:
            # Submitted without a place: under the root, named by what it computes.
            resolved = composition._resolve_paths(model, base)
            target = self.root / "runs" / f"{model.run.name}-{run_id(resolved)[:SHORT]}"
            model = model.model_copy(
                update={"run": model.run.model_copy(update={"output_dir": target})}
            )
        return composition._resolve_paths(model, base)

    def _placed(self, config: HistoryMatchConfig) -> HistoryMatchConfig:
        """Return ``config`` with a missing output directory set to where the root puts it."""
        if config.run.output_dir is not None:
            return config
        target = self.root / "runs" / config.run.name
        return config.model_copy(
            update={"run": config.run.model_copy(update={"output_dir": target})}
        )

    def validate(
        self,
        config: HistoryMatchConfig | Mapping[str, Any] | Path,
        base_dir: Path | None = None,
        engine: EngineMethod | None = None,
        *,
        check_output: bool = False,
        place_in: Path | None = None,
        annotations: Annotations | None = None,
    ) -> Validation:
        """Validate a configuration and plan its cost; nothing is simulated.

        The service's ``[policy]`` is checked too: a violation makes the
        configuration invalid, with the reasons among the errors. With ``engine``
        the case runs on another engine: that engine checks it and plans its cost.

        With ``check_output`` the run's folder is checked as submitting it would be
        (:meth:`output_conflict`): one holding another run is an error, so that a validation
        agrees with the submission (the REST API, the client, the tools and ``apex
        validate`` ask for it). ``place_in`` places the run as the web UI does
        (``<place_in>/runs/<name>``), and ``annotations`` names its submitter.
        Off by default: callers that validate what is not to be submitted as it is (a
        study's cells, a diagnosis' patch, a submission's own check) leave it out.
        """
        return self._checked_output(
            self._engine_validation(config, base_dir, engine),
            config, base_dir, engine, check_output, place_in, annotations,
        )  # fmt: skip

    def _checked_output(
        self,
        validation: Validation,
        config: HistoryMatchConfig | Mapping[str, Any] | Path,
        base_dir: Path | None,
        engine: EngineMethod | None,
        check: bool,
        place_in: Path | None,
        annotations: Annotations | None,
    ) -> Validation:
        """Add :meth:`output_conflict`'s error to a valid validation, if ``check`` finds one."""
        if not check or not validation.valid or validation.plan is None:
            return validation
        try:
            resolved = self.resolve(config, base_dir)
        except (ApexError, ValueError):
            return validation  # the validation would have said
        if place_in is not None:
            resolved = resolved.model_copy(update={"run": resolved.run.model_copy(
                update={"output_dir": place_in / "runs" / resolved.run.name})})  # fmt: skip
        engine_json = None if engine is None else engine.as_json()
        conflict = self.output_conflict(resolved, annotations, engine_json)
        if conflict is None:
            return validation
        return validation.model_copy(
            update={"valid": False, "errors": [*validation.errors, conflict]}
        )

    def _engine_validation(
        self,
        config: HistoryMatchConfig | Mapping[str, Any] | Path,
        base_dir: Path | None,
        engine: EngineMethod | None,
    ) -> Validation:
        validation = self._validate(config, base_dir)
        if engine is None or not validation.valid or validation.plan is None:
            return validation
        if policy.active(self.jobs.settings.policy):
            # Another engine runs programs of its own (ERT, its forward-model steps) that the
            # policy cannot check: refused rather than let through unchecked.
            return Validation(
                valid=False,
                errors=[f"the {engine.engine} engine is not allowed under a service [policy]"],
            )
        try:
            runner = engine_plugin(engine.engine)
        except ApexError as error:
            return Validation(valid=False, errors=[str(error)])
        resolved = self.resolve(config, base_dir)
        errors = runner.check(resolved, engine)
        if errors:
            return Validation(valid=False, errors=errors)
        plan = validation.plan.model_copy(
            update={
                "run_id": run_id(resolved, engine.as_json()),
                "method": f"{engine.engine}:{engine.name}",
                "engine": engine.engine,
                "max_simulator_runs": runner.simulator_runs(engine, validation.plan.members),
                "max_forecasts": None,
            }
        )
        return Validation(valid=True, plan=plan)

    def _validate(
        self, config: HistoryMatchConfig | Mapping[str, Any] | Path, base_dir: Path | None = None
    ) -> Validation:
        if policy.active(self.jobs.settings.policy):
            # Before validation reads any file or builds any plugin.
            try:
                self.resolve(config, base_dir)
            except policy.PolicyViolation as error:
                return Validation(valid=False, errors=[str(error)])
            except ConfigError:
                pass  # not a policy matter: validation reports it below
        return validate_history_match(config, base_dir=base_dir, store=self.store)

    # ------------------------------------------------------------------ work

    def gate(
        self,
        run_ids: Sequence[str],
        simulator_runs: int | None,
        *,
        approve: "Callable[[dict[str, Any]], bool] | None" = None,
        approval_token: str | None = None,
        client: str = "user",
        annotations: Mapping[str, str] | None = None,
    ) -> None:
        """Let a submission through, or refuse it until a person approves.

        Approval comes from ``approval_token``, from ``approve`` (an interactive yes), or
        from a grant of ``client`` for the ``experiment`` in ``annotations``.

        Raises
        ------
        ApprovalRequired
            With a pending token, when approval is needed and not given.
        """
        quota = self.jobs.settings.quotas.get(client)
        limit = None if quota is None else quota.max_simulator_runs_per_day
        if limit is not None and simulator_runs is not None and simulator_runs > limit:
            raise ConfigError(
                f"this submission may take {simulator_runs} simulator runs, more than client "
                f"{client!r} may use in a day ({limit}, service.toml [quotas.{client}])"
            )
        if not self.approvals.needed(simulator_runs, len(run_ids)):
            return
        print_ = fingerprint(run_ids)
        cost = {"simulator_runs": simulator_runs, "runs": len(run_ids)}
        if approval_token is not None:
            self.approvals.consume(approval_token, print_)
            return
        declined = False
        if approve is not None:
            if approve(cost):
                self.approvals.record(print_, cost, by=f"{client}: interactive")
                return
            declined = True  # a person's explicit no is final: no grant overrides it
        experiment = (annotations or {}).get("experiment")
        if not declined:
            granted = self.grants.charge(
                client,
                experiment,
                simulator_runs,
                record=lambda grant: self.approvals.record(print_, cost, by=f"grant {grant}"),
            )
            if granted is not None:
                return
        token = self.approvals.request(print_, cost, requested_by=client)
        limits = self.approvals.settings
        why = "" if declined else self.grants.why_not(client, experiment, simulator_runs)
        reason = (
            f"this submission may take {simulator_runs} simulator runs in {len(run_ids)} "
            f"runs, which need approval (limits: {limits.max_simulator_runs} simulator runs, "
            f"{limits.max_runs} runs)" + (f" (no grant applies: {why})" if why else "")
        )
        raise ApprovalRequired(
            f"{reason}; run `apex approve {token}`, then submit again with approval_token={token}",
            token,
            reason,
        )

    def reservation(self, config: HistoryMatchConfig, parallel_runs: int | None) -> dict[str, int]:
        """Return a run's machine-wide reservation: runs at once x each run's tokens."""
        return self._reservation(config.simulator, parallel_runs)

    def _reservation(self, section: PluginSection, parallel_runs: int | None) -> dict[str, int]:
        try:
            simulator = simulators.create(section.name, **section.options)
        except (TypeError, ValueError, ApexError):
            return {}
        needs: dict[str, int] = dict(getattr(simulator, "resources", None) or {})
        cap = getattr(simulator, "max_parallel_runs", None)
        at_once = max(1, min(parallel_runs or 1, cap or parallel_runs or 1))
        return {pool: n * at_once for pool, n in needs.items()}

    def submit_run(
        self,
        config: HistoryMatchConfig | Mapping[str, Any] | Path,
        *,
        annotations: Annotations | None = None,
        base_dir: Path | None = None,
        max_steps: int | None = None,
        force_new: bool = False,
        parent: str | None = None,
        priority: str = "normal",
        client: str = "user",
        approve: "Callable[[dict[str, Any]], bool] | None" = None,
        approval_token: str | None = None,
        approved: bool = False,
        engine: EngineMethod | None = None,
        code: str | None = None,
    ) -> JobHandle:
        """Run a configuration in the background; an identical run is reused.

        ``priority`` (``high``, ``normal``, ``low``) and ``client`` (for quotas) go to the
        scheduler. ``approved`` is for callers that approved the whole
        submission already (a study). ``engine`` runs the case on another engine.
        ``code`` is the code it runs (:meth:`pin_code`).

        Raises
        ------
        ConfigError
            If the configuration is invalid, or its output directory holds another run.
        ApprovalRequired
            If it needs a person's approval first.
        """
        resolved = self.resolve(config, base_dir)  # checks the policy first
        # Its own check of the run folder follows, after the reuse it allows.
        validation = self.validate(resolved, engine=engine, check_output=False)
        if not validation.valid or validation.plan is None:
            raise ConfigError("invalid configuration: " + "; ".join(validation.errors))
        plan = validation.plan
        engine_json = None if engine is None else engine.as_json()
        # What approvals are bound to (validate's plan.run_id).
        content = run_id(resolved, engine_json)
        identity = owned_run_id(content, (annotations or {}).get(SUBMITTED_BY))
        existing = None if force_new else self.store.find(identity)

        def gate() -> None:
            self.gate(
                [content],
                plan.max_simulator_runs,
                approve=approve,
                approval_token=approval_token,
                client=client,
                annotations=annotations,
            )

        # A run already complete, queued or running is reused and runs nothing new, so it
        # needs no approval (release tester N8, 2026-10-20: an identical run asked to
        # approve thousands of simulator runs that would not run). Should the job it found
        # end before _reuse looks again, _reuse gates the run it then starts.
        free = existing is not None and self._free_reuse(identity, existing.path)
        if existing is None and not force_new:
            # Queued, its folder not made yet (the catalogue finds runs by their folders):
            # the job is the run, so it is returned rather than submitted twice.
            live = self._live_job(identity)
            if live is not None:
                return JobHandle(id=live["id"], kind=live["kind"], run_ids=[identity],
                                 reused=True)  # fmt: skip
        if not approved and not free:
            gate()
        scheduling: dict[str, Any] = {
            "priority": priority,
            "client": client,
            "reservation": self.reservation(resolved, plan.parallel_runs),
            "planned_runs": plan.max_simulator_runs or 0,
        }
        if existing is not None:
            # A reused run resumes from its own snapshot: it must be one the service wrote.
            policy.check_source(existing.path, self.jobs.settings.policy, self.root)
            return self._reuse(
                identity, existing.path, annotations, max_steps, parent, scheduling, resolved,
                engine_json, code, gate=gate if free and not approved else None,
            )  # fmt: skip
        assert resolved.run.output_dir is not None
        output = resolved.run.output_dir
        if (output / "state.json").exists():
            # The same check as output_conflict's, which the REST API's validation makes.
            raise ConfigError(OUTPUT_TAKEN.format(output=output))
        # Frozen only now that a job will run (a freeze may take a minute).
        scheduling["code"] = self.pin_code(code)
        self.store.register(
            CatalogueEntry(
                identity,
                output,
                resolved.run.name,
                now(),
                {
                    **(annotations or {}),
                    "case_key": case_key(resolved),
                    **({"engine": engine.engine} if engine is not None else {}),
                },
            )
        )
        job = self.jobs.submit(
            "run",
            {
                "config": resolved.model_dump_json(),
                "max_steps": max_steps,
                **({"engine": engine_json} if engine_json is not None else {}),
            },
            run_ids=[identity],
            run_dirs=[str(output)],
            annotations=annotations,
            parent=parent,
            **scheduling,
        )
        return JobHandle(id=job, kind="run", run_ids=[identity])

    def free_reuse(
        self, config: HistoryMatchConfig, annotations: Annotations | None = None
    ) -> bool:
        """Whether submitting the resolved ``config`` would reuse a run and simulate nothing.

        As :meth:`submit_run` decides it (an identical run complete, queued or running):
        the tools issue no pending token and charge no grant for it.
        """
        identity = owned_run_id(run_id(config, None), (annotations or {}).get(SUBMITTED_BY))
        existing = self.store.find(identity)
        if existing is None:
            return self._live_job(identity) is not None
        return self._free_reuse(identity, existing.path)

    def _live_job(self, identity: str) -> dict[str, Any] | None:
        """Return the last queued or running job of the run ``identity``, if there is one."""
        for record in reversed(self.jobs.all_records()):
            if identity in record["run_ids"] and record["state"] not in FINAL:
                return record
        return None

    def _free_reuse(self, identity: str, path: Path) -> bool:
        """Whether reusing the run ``identity`` at ``path`` simulates nothing (see ``_reuse``).

        True when a job of it is still queued or running (that job is returned) or the run
        is complete; a run to start again or resume spends simulator runs.
        """
        return self._live_job(identity) is not None or _complete(path)

    def output_conflict(
        self,
        config: HistoryMatchConfig,
        annotations: Annotations | None = None,
        engine: dict[str, Any] | None = None,
    ) -> str | None:
        """Return why submitting ``config`` (resolved, placed) would be refused, if it would.

        Its output folder holds another run, and no identical run (of the same submitter:
        ``owned_run_id``) is in the catalogue to be reused instead: :meth:`submit_run`'s
        check (without ``force_new``; ``engine`` as :meth:`EngineMethod.as_json` gives it).
        ``None`` if the submission would run or reuse. :meth:`validate` reports it with
        ``check_output``, so that Validate and Submit agree.
        """
        output = config.run.output_dir
        if output is None or not (output / "state.json").exists():
            return None
        identity = owned_run_id(run_id(config, engine), (annotations or {}).get(SUBMITTED_BY))
        if self.store.find(identity) is not None or self._live_job(identity) is not None:
            return None  # reused, as submit_run would
        return OUTPUT_TAKEN.format(output=output)

    def _reuse(
        self,
        identity: str,
        path: Path,
        annotations: Annotations | None,
        max_steps: int | None,
        parent: str | None,
        scheduling: dict[str, Any],
        config: HistoryMatchConfig,
        engine: dict[str, Any] | None = None,
        code: str | None = None,
        gate: Callable[[], None] | None = None,
    ) -> JobHandle:
        """Reuse the run ``identity`` at ``path``: its live job, its result, or a new job.

        ``gate`` approves a submission that was let through as a free reuse; it is called
        before a job that simulates (a restart or a resume) is queued.
        """
        records = self.jobs.all_records()
        for record in reversed(records):
            if identity in record["run_ids"] and record["state"] not in FINAL:
                return JobHandle(
                    id=record["id"], kind=record["kind"], run_ids=[identity], reused=True
                )
        # Where this submission asked to run: the owner's web UI shows its job by that
        # folder (``OwnerView``) when the run lies elsewhere (a study's cell, the CLI's
        # run), whether the job reuses it as is, restarts or resumes it.
        requested = config.run.output_dir
        asked = {"requested_dir": str(requested)} if requested is not None else {}
        if _complete(path):
            # Recorded under the submitting client too (the web UI's users see their own
            # jobs by it). The spec records the run it reused, and the code that
            # made it: this job runs no code of its own, live or frozen.
            made = run_commit(Path(path)) or _made_with(records, identity)
            job = self.jobs.submit(
                "run",
                {"reused": str(path), **asked, **({"reused_code": made} if made else {})},
                run_ids=[identity], run_dirs=[str(path)],
                annotations=annotations, parent=parent, state="complete",
                client=scheduling.get("client", "user"),
            )  # fmt: skip
            return JobHandle(id=job, kind="run", run_ids=[identity], reused=True)
        if engine is not None or not (Path(path) / "state.json").exists():
            # Another engine has no resume: an unfinished run starts again.
            # It failed (or was stopped) before its first checkpoint: there is nothing to
            # resume, so it runs again from the start, in its own folder (a client's report,
            # 2026-09-29: the resume failed at once with "no saved state").
            placed = config.model_copy(
                update={"run": config.run.model_copy(update={"output_dir": Path(path)})}
            )
            if gate is not None:
                gate()
            job = self.jobs.submit(
                "run",
                {
                    "config": placed.model_dump_json(),
                    "max_steps": max_steps,
                    **({"engine": engine} if engine is not None else {}),
                    **asked,
                },
                run_ids=[identity], run_dirs=[str(path)], annotations=annotations,
                parent=parent, **scheduling, code=self.pin_code(code),
            )  # fmt: skip
            return JobHandle(id=job, kind="run", run_ids=[identity], reused=False)
        if gate is not None:
            gate()
        # A stopped run continues on the code it ran, unless the caller says otherwise.
        pin = self._resume_pin(code, path, [r for r in records if identity in r["run_ids"]])
        job = self.jobs.submit(
            "resume", {"run_dir": str(path), "max_steps": max_steps, **asked},
            run_ids=[identity],
            run_dirs=[str(path)], annotations=annotations, parent=parent, **scheduling,
            code=pin,
        )  # fmt: skip
        return JobHandle(id=job, kind="resume", run_ids=[identity], reused=True)

    def _jobs_of(self, run_dir: Path) -> list[dict[str, Any]]:
        """Return the jobs that ran (or reused) ``run_dir``, oldest first."""
        where = str(run_dir)
        return [r for r in self.jobs.all_records() if where in r["run_dirs"]]

    def _resume_pin(
        self, code: str | None, run_dir: Path, records: Sequence[dict[str, Any]]
    ) -> dict[str, str] | None:
        """Return a resume's pin: ``code``'s, else the one the run's last job had.

        A run submitted frozen and resumed without naming the code (the web UI's button,
        ``Client.resume``) keeps its commit: never a mix of versions.
        """
        if code is not None:
            return self.pin_code(code, run_dir)
        for record in reversed(records):
            pin = record.get("code")
            if pin:
                return self.pin_code(str(pin["commit"]))
            if record.get("state") != "complete" or "reused" not in record.get("spec", {}):
                break  # the last job that ran it was live
        return None

    def submit_prediction(
        self,
        config: PredictionConfig | Path,
        *,
        annotations: Annotations | None = None,
        after: Sequence[str] = (),
        parent: str | None = None,
        priority: str = "normal",
        client: str = "user",
        code: str | None = None,
    ) -> JobHandle:
        """Forecast a run beyond history in the background.

        ``code`` is the code it runs (:meth:`pin_code`): ``"frozen"`` is the run's own.
        """
        rules = self.jobs.settings.policy
        if isinstance(config, Path):
            policy.check_file(config, rules, self.root)  # before it is opened
        resolved = prediction.load_prediction_config(config) if isinstance(config, Path) else config
        policy.check(resolved, rules, self.root)  # run_dir must be a run the service wrote
        assert resolved.prediction.output_dir is not None
        reservation: dict[str, int] = {}
        try:
            run_config = composition.snapshot_config(RunDirectory(resolved.prediction.run_dir))
        except ConfigError:
            run_config = None  # the run does not exist yet (a study's prediction waits for it)
        if run_config is not None:
            simulator = prediction.simulator_section(resolved, run_config)
            # What will actually run: the run's snapshot with the prediction's override.
            policy.check(run_config.model_copy(update={"simulator": simulator}), rules, self.root)
            execution = resolved.execution or run_config.execution
            workers = execution.options.get("workers")
            reservation = self.reservation(
                run_config.model_copy(update={"simulator": simulator}),
                int(workers) if workers else None,
            )
        job = self.jobs.submit(
            "prediction",
            {"config": resolved.model_dump_json()},
            run_dirs=[str(resolved.prediction.output_dir)],
            annotations=annotations,
            after=after,
            parent=parent,
            priority=priority,
            client=client,
            reservation=reservation,
            code=self.pin_code(code, resolved.prediction.run_dir),
        )
        return JobHandle(id=job, kind="prediction")

    def status(self, job: str) -> JobStatus:
        """Return where a job is."""
        return self.jobs.status(job)

    def events(self, job: str, *, since: int = 0) -> list[Event]:
        """Return a job's events from number ``since`` on."""
        return self.jobs.events(job, since)

    def stop(self, job: str) -> None:
        """Stop a job: queued ones are cancelled, running runs stop at their next checkpoint."""
        self.jobs.stop(job)

    def resume(
        self, job: str, *, max_steps: int | None = None, code: str | None = None
    ) -> JobHandle:
        """Continue a stopped run or optimization job (``max_steps``: accepted iterations).

        ``code`` is the code it runs (:meth:`pin_code`): ``"frozen"`` is the run's own.
        """
        record = self.jobs.record(job)
        if record["kind"] in ("optimization", "optimization_resume"):
            return self._resume_optimization(record, max_steps, code)
        if record["kind"] not in ("run", "resume") or not record["run_dirs"]:
            raise ConfigError(f"job {job} is not a history-matching run")
        rules = self.jobs.settings.policy
        run_dir = Path(record["run_dirs"][0])
        if not (run_dir / "state.json").exists() and in_use(run_dir):
            raise ConfigError(
                f"job {job}'s run in {run_dir} is in use by another process, before its "
                "first checkpoint; wait for it, or stop it, before resuming it"
            )
        if not (run_dir / "state.json").exists():
            # Stopped (or failed) before its first step ended: nothing to continue.
            raise ConfigError(
                f"job {job}'s run in {run_dir} has no checkpoint to resume from: it ended "
                "before its first step did; submit its configuration again"
            )
        policy.check_source(run_dir, rules, self.root)
        if policy.active(rules):
            policy.check(composition.snapshot_config(RunDirectory(run_dir)), rules, self.root)
        new = self.jobs.submit(
            "resume",
            {
                "run_dir": record["run_dirs"][0],
                "max_steps": max_steps,
                # The owner's web UI still shows a reuse's job by it (OwnerView).
                **{k: v for k, v in (record.get("spec") or {}).items() if k == "requested_dir"},
            },
            run_ids=record["run_ids"],
            run_dirs=record["run_dirs"],
            annotations=record["annotations"],
            priority=record.get("priority", "normal"),
            client=record.get("client", "user"),
            reservation=record.get("reservation") or {},
            planned_runs=record.get("planned_runs") or 0,
            code=self._resume_pin(code, run_dir, self._jobs_of(run_dir)),
        )
        return JobHandle(id=new, kind="resume", run_ids=record["run_ids"])

    def resume_run(
        self,
        run_dir: Path,
        *,
        max_steps: int | None = None,
        code: str | None = None,
        priority: str = "normal",
        client: str = "user",
    ) -> JobHandle:
        """Continue the run in ``run_dir`` in the background (``apex submit --resume``).

        A run the service knows resumes as :meth:`resume` of its last job (a job still
        going is returned as it is); a run started outside it (``apex run``) gets a resume
        job of its own. ``code`` as :meth:`pin_code`: ``"frozen"`` is the run's own commit.

        Raises
        ------
        ConfigError
            If ``run_dir`` holds no resumable run.
        """
        path = Path(run_dir).resolve()
        for record in reversed(self.jobs.all_records()):
            if record["kind"] in ("prediction", "study") or path not in {
                Path(d).resolve() for d in record["run_dirs"]
            }:
                continue
            if record["state"] not in FINAL:
                return JobHandle(
                    id=record["id"], kind=record["kind"], run_ids=record["run_ids"], reused=True
                )
            return self.resume(record["id"], max_steps=max_steps, code=code)
        rules = self.jobs.settings.policy
        if not (path / "state.json").exists():
            if (path / opt_layout.CHECKPOINT_VALUES).exists():
                # An optimization started by apex optimize.
                policy.check_source(path, rules, self.root)
                job = self.jobs.submit(
                    "optimization_resume",
                    {"run_dir": str(path), "max_iterations": max_steps},
                    run_dirs=[str(path)], priority=priority, client=client,
                    code=self._resume_pin(code, path, []),
                )  # fmt: skip
                return JobHandle(id=job, kind="optimization_resume")
            raise ConfigError(f"{path} holds no run or optimization to resume")
        policy.check_source(path, rules, self.root)
        snapshot = composition.snapshot_config(RunDirectory(path))
        if policy.active(rules):
            policy.check(snapshot, rules, self.root)
        workers = snapshot.execution.options.get("workers")
        job = self.jobs.submit(
            "resume",
            {"run_dir": str(path), "max_steps": max_steps},
            run_dirs=[str(path)],
            priority=priority,
            client=client,
            reservation=self.reservation(snapshot, int(workers) if workers else None),
            code=self._resume_pin(code, path, []),
        )
        return JobHandle(id=job, kind="resume")

    # ------------------------------------------------------------ optimization

    def resolve_optimization(
        self,
        config: OptimizationConfig | Mapping[str, Any] | Path,
        base_dir: Path | None = None,
    ) -> OptimizationConfig:
        """Resolve an optimization configuration and place it, policy-checked.

        An unset ``[optimization] output_dir`` becomes ``<root>/optimizations/<name>/<id>``.

        Raises
        ------
        ConfigError
            If it is invalid.
        PolicyViolation
            If it breaks the service's ``[policy]``.
        """
        from apex.services import optimization as opt
        from apex.services.identity import optimization_id

        rules = self.jobs.settings.policy
        if isinstance(config, Path):
            policy.check_file(config, rules, self.root)
            resolved = opt.load_optimization_config(config, place=False)
        else:
            resolved = opt.resolve_optimization_config(config, base_dir or Path.cwd(), place=False)
        if resolved.optimization.output_dir is None:
            place = (
                self.root
                / opt_layout.SERVICE_FOLDER
                / resolved.optimization.name
                / optimization_id(resolved)[:12]
            )
            resolved = resolved.model_copy(
                update={"optimization": resolved.optimization.model_copy(
                    update={"output_dir": place}
                )}
            )  # fmt: skip
        policy.check(resolved, rules, self.root)
        return resolved

    def validate_optimization(
        self,
        config: OptimizationConfig | Mapping[str, Any] | Path,
        base_dir: Path | None = None,
    ) -> Validation:
        """Validate an optimization configuration and plan its cost; nothing is evaluated."""
        from apex.services import optimization as opt

        try:
            resolved = self.resolve_optimization(config, base_dir)
            plan = opt.plan_optimization(resolved)
        except (ApexError, ValueError, TypeError) as error:
            return Validation(valid=False, errors=[str(error)])
        return Validation(valid=True, optimization=plan)

    def submit_optimization(
        self,
        config: OptimizationConfig | Mapping[str, Any] | Path,
        *,
        base_dir: Path | None = None,
        max_iterations: int | None = None,
        annotations: Annotations | None = None,
        after: Sequence[str] = (),
        parent: str | None = None,
        priority: str = "normal",
        client: str = "user",
        approve: "Callable[[dict[str, Any]], bool] | None" = None,
        approval_token: str | None = None,
        approved: bool = False,
        code: str | None = None,
    ) -> JobHandle:
        """Optimize in the background; the approval rule applies as for runs.

        ``code`` is the code it runs (:meth:`pin_code`).

        Raises
        ------
        ConfigError
            If the configuration is invalid.
        ApprovalRequired
            If its evaluation bound needs a person's approval first.
        """
        from apex.services import optimization as opt

        resolved = self.resolve_optimization(config, base_dir)
        try:
            plan = opt.plan_optimization(resolved)
        except (ValueError, TypeError) as error:
            raise ConfigError(f"invalid optimization: {error}") from error
        if not approved:
            self.gate(
                [plan.optimization_id],
                plan.max_simulator_runs,
                approve=approve,
                approval_token=approval_token,
                client=client,
                annotations=annotations,
            )
        assert resolved.optimization.output_dir is not None  # placed above
        reservation = (
            {}
            if resolved.simulator is None
            else self._reservation(resolved.simulator, plan.parallel_runs)
        )
        job = self.jobs.submit(
            "optimization",
            {"config": resolved.model_dump_json(), "max_iterations": max_iterations},
            run_ids=[plan.optimization_id],
            run_dirs=[str(resolved.optimization.output_dir)],
            annotations=annotations,
            after=after,
            parent=parent,
            priority=priority,
            client=client,
            reservation=reservation,
            planned_runs=plan.max_simulator_runs or 0,
            code=self.pin_code(code),
        )
        return JobHandle(id=job, kind="optimization", run_ids=[plan.optimization_id])

    def _resume_optimization(
        self, record: dict[str, Any], max_iterations: int | None, code: str | None = None
    ) -> JobHandle:
        run_dir = Path(record["run_dirs"][0])
        rules = self.jobs.settings.policy
        policy.check_source(run_dir, rules, self.root)
        new = self.jobs.submit(
            "optimization_resume",
            {"run_dir": str(run_dir), "max_iterations": max_iterations},
            run_ids=record["run_ids"],
            run_dirs=record["run_dirs"],
            annotations=record["annotations"],
            priority=record.get("priority", "normal"),
            client=record.get("client", "user"),
            reservation=record.get("reservation") or {},
            planned_runs=record.get("planned_runs") or 0,
            code=self._resume_pin(code, run_dir, self._jobs_of(run_dir)),
        )
        return JobHandle(id=new, kind="optimization_resume", run_ids=record["run_ids"])

    def optimizations(self) -> list[dict[str, Any]]:
        """Return the optimization jobs: id, state, run directory, name and times."""
        rows = []
        for record in self.jobs.all_records():
            if record["kind"] not in ("optimization", "optimization_resume"):
                continue
            rows.append({
                "job": record["id"],
                "kind": record["kind"],
                "state": record["state"],
                "path": (record.get("run_dirs") or [None])[0],
                "created": record.get("created"),
                "finished": record.get("finished"),
                "client": record.get("client"),
                "annotations": record.get("annotations") or {},
            })  # fmt: skip
        return rows

    def optimization_summary(self, path: Path) -> "OptimizationSummary":
        """Return an optimization run's progress and result."""
        from apex.services import optimization as opt

        policy.check_source(path, self.jobs.settings.policy, self.root)
        return opt.optimization_summary(path)

    def wait(self, job: str, *, timeout_s: float | None = None, poll_s: float = 1.0) -> JobStatus:
        """Block until a job ends (for scripts and tests); return its final status."""
        import time

        began = time.monotonic()
        while True:
            status = self.status(job)
            if status.state in FINAL:
                return status
            if timeout_s is not None and time.monotonic() - began > timeout_s:
                return status
            time.sleep(poll_s)

    # ------------------------------------------------------------------ studies

    def _study(self, spec: "StudySpec | Path", base_dir: Path | None) -> tuple["StudySpec", Path]:
        from apex.services.study import load_study, resolve_case

        rules = self.jobs.settings.policy
        # Case paths may be relative to the workspace, as the UI shows them.
        space = self.workspace()
        if isinstance(spec, Path):
            policy.check_file(spec, rules, self.root)  # before it is opened
            study, base = load_study(spec, space), spec.resolve().parent
        else:
            study, base = spec, (base_dir or Path.cwd()).resolve()
        # Each case file, resolved against the study's base once, checked and then used
        # under that same absolute path (a relative path would otherwise be opened against
        # the service's working directory: code review, delta round 1).
        cases = {name: resolve_case(case, base, space) for name, case in study.study.cases.items()}
        for case in cases.values():
            policy.check_file(case, rules, self.root)  # before it is opened
        study = study.model_copy(update={"study": study.study.model_copy(update={"cases": cases})})
        return study, base

    def checked_optimization_study(
        self, path: "Path | OptimizationStudySpec", base_dir: Path | None = None
    ) -> "tuple[OptimizationStudySpec, Path] | None":
        """Load a study of optimizations as :meth:`checked_study` does; ``None`` for another.

        ``path`` is its file, or its specification with ``base_dir`` for its cases.

        Raises
        ------
        PolicyViolation
            If the study file or a case file lies outside the policy's roots.
        """
        return self._optimization_study(path, base_dir)

    def checked_study(self, path: Path) -> tuple["StudySpec", Path]:
        """Load a study file, with its cases' paths absolute and checked by the ``[policy]``.

        For callers that plan a study themselves (the tool layer): the study file and each
        case file are checked before they are opened, as :meth:`validate_study` does.

        Raises
        ------
        PolicyViolation
            If the study file or a case file lies outside the policy's roots.
        """
        return self._study(path, None)

    def _optimization_study(
        self, spec: "OptimizationStudySpec | StudySpec | Path", base_dir: Path | None
    ) -> "tuple[OptimizationStudySpec, Path] | None":
        """Return a study of optimizations, cases resolved and checked; else None."""
        import tomllib

        from apex.domain.optimization_study import OptimizationStudySpec
        from apex.services.optimization_study import (
            is_optimization_study,
            load_optimization_study,
        )
        from apex.services.study import resolve_case

        rules = self.jobs.settings.policy
        space = self.workspace()
        if isinstance(spec, Path):
            policy.check_file(spec, rules, self.root)  # before it is opened
            try:
                document = tomllib.loads(spec.read_text("utf-8"))
            except (OSError, tomllib.TOMLDecodeError):
                return None  # the history-matching loader names the problem
            if not is_optimization_study(document):
                return None
            study, base = load_optimization_study(spec, space), spec.resolve().parent
        elif isinstance(spec, OptimizationStudySpec):
            study, base = spec, (base_dir or Path.cwd()).resolve()
        else:
            return None
        cases = {n: resolve_case(c, base, space) for n, c in study.study.cases.items()}
        for case in cases.values():
            policy.check_file(case, rules, self.root)
        return study.model_copy(
            update={"study": study.study.model_copy(update={"cases": cases})}
        ), base

    def validate_study(
        self, spec: "StudySpec | OptimizationStudySpec | Path", base_dir: Path | None = None
    ) -> Validation:
        """Expand a study and validate every run; check its budget. Nothing is simulated."""
        from apex.services.study import plan_study

        optimizing = self._optimization_study(spec, base_dir)
        if optimizing is not None:
            from apex.services.optimization_study import validate_optimization_study

            return validate_optimization_study(self, *optimizing)
        study, base = self._study(spec, base_dir)  # type: ignore[arg-type]
        plan = plan_study(self, study, base)
        notes = [f"not run: {c.label}: {c.reason}" for c in plan.skipped]
        return plan.validation.model_copy(update={"notes": notes})

    def submit_study(
        self,
        spec: "StudySpec | OptimizationStudySpec | Path",
        *,
        base_dir: Path | None = None,
        approve: "ApprovalHook | None" = None,
        approval_token: str | None = None,
        priority: str = "normal",
        client: str = "user",
        approved: bool = False,
        output_root: Path | None = None,
        check: "Callable[[Any], None] | None" = None,
        code: str | None = None,
    ) -> JobHandle:
        """Run a study's cells (and predictions) as jobs; identical runs are reused.

        ``approved`` is for callers that checked approval themselves (the tool layer).
        ``code`` is the code every cell runs (:meth:`study_code`).
        ``output_root`` holds the cells' runs (default ``<root>/studies``; the web UI's
        users' folders). ``check`` is called with every cell's configuration as
        planned for submission, and may raise to refuse the study (the web UI's guard: what
        it checks is what runs, not files read again).

        Raises
        ------
        ConfigError
            If the study is invalid or over its budget.
        ApprovalRequired
            If it needs a person's approval (``approve`` said no, or none was given).
        Exception
            Whatever ``check`` raises, before anything is submitted.
        """
        from apex.services.study import submit_study

        optimizing = self._optimization_study(spec, base_dir)
        if optimizing is not None:
            from apex.services.optimization_study import submit_optimization_study

            return submit_optimization_study(
                self, *optimizing, approve=approve, approval_token=approval_token,
                priority=priority, client=client, approved=approved, output_root=output_root,
                check=check, code=code,
            )  # fmt: skip
        study, base = self._study(spec, base_dir)  # type: ignore[arg-type]
        return submit_study(
            self,
            study,
            base,
            approve=approve,
            approval_token=approval_token,
            priority=priority,
            client=client,
            approved=approved,
            output_root=output_root,
            check=check,
            code=code,
        )

    def study_status(self, name: str) -> JobStatus:
        """Return a study's status: its job, with every cell's state."""
        from apex.services.study import study_record

        return self.status(str(study_record(self, name)["job"]))

    def compare(self, study: str, *, metrics: Sequence[str] | None = None) -> "Comparison":
        """Compare a study's variants at each run's posterior (``comparison.json``/``.csv``).

        A study of optimizations compares final and best objectives instead.
        """
        from apex.services.study import compare_study, study_record

        if study_record(self, study).get("kind") == "optimization":
            from apex.services.optimization_study import compare_optimization_study

            return compare_optimization_study(self, study, list(metrics) if metrics else None)
        return compare_study(self, study, metrics)

    # ------------------------------------------------------------------ results

    def run_summary(self, run: str | Path) -> RunSummary:
        """Return a run's results, by id or by directory.

        Raises
        ------
        ConfigError
            If there is no such run.
        """
        path = Path(run) if isinstance(run, Path) else None
        entry = None
        if path is not None:
            policy.check_source(path, self.jobs.settings.policy, self.root)
        if path is None:
            entry = self.store.find(str(run))
            if entry is None:
                raise ConfigError(f"no run {run!r} in {self.root}")
            path = entry.path
        return summarise_run(path, entry.annotations if entry else {})

    def run_steps(self, run: str | Path) -> RunSteps:
        """Return a run's performance metrics step by step, by id or by directory.

        A prediction directory (``RUN/predictions/NAME``) gives one step per ensemble.

        Raises
        ------
        ConfigError
            If there is no such run, or it has nothing to summarise yet.
        """
        if isinstance(run, Path):
            policy.check_source(run, self.jobs.settings.policy, self.root)
            path = run
        else:
            entry = self.store.find(str(run))
            if entry is None:
                raise ConfigError(f"no run {run!r} in {self.root}")
            path = entry.path
        from apex.services.summaries import summarize_one

        return steps_document(summarize_one(path))

    def workspace(self) -> "Workspace":
        """Return the editor's workspace."""
        from apex.services.workspace import Workspace

        return Workspace(self.root, self.jobs.settings)

    def validate_text(
        self,
        text: str,
        base_dir: Path,
        *,
        check_output: bool = False,
        place_in: Path | None = None,
        annotations: Annotations | None = None,
    ) -> Validation:
        """Validate a configuration given as TOML text (a run's or a study's).

        Relative paths resolve against ``base_dir``; nothing is saved or simulated. The
        keywords are :meth:`validate`'s, for a run's configuration.
        """
        import tomllib

        try:
            document = tomllib.loads(text)
        except tomllib.TOMLDecodeError as error:
            return Validation(valid=False, errors=[f"TOML: {error}"])
        if "study" in document:
            from apex.domain.optimization_study import OptimizationStudySpec
            from apex.domain.study import StudySpec
            from apex.services.optimization_study import is_optimization_study

            model = OptimizationStudySpec if is_optimization_study(document) else StudySpec
            try:
                spec = model.model_validate(document)
            except ValidationError as error:
                return Validation(valid=False, errors=[validation_problems(error)])
            return self.validate_study(spec, base_dir=base_dir)
        if "optimization" in document and "optimizer" in document:
            return self.validate_optimization(document, base_dir)
        return self.validate(document, base_dir=base_dir, check_output=check_output,
                             place_in=place_in, annotations=annotations)  # fmt: skip

    def prediction_summary(self, path: Path) -> PredictionSummary:
        """Return a finished prediction's results."""
        policy.check_source(path, self.jobs.settings.policy, self.root)
        return summarise_prediction(path)


def _made_with(records: list[dict[str, Any]], identity: str) -> str | None:
    """Return the commit of the job that made run ``identity`` (or "live checkout"), if any.

    The fallback of the run's own provenance (:func:`run_commit`), which names no commit
    for a run started from an uncommitted tree.
    """
    for record in reversed(records):
        spec = record.get("spec") or {}
        if identity not in record["run_ids"] or record["state"] != "complete":
            continue
        if "reused" in spec:
            made = spec.get("reused_code")
            if isinstance(made, str):
                return made
            continue
        return str((record.get("code") or {}).get("commit") or "live checkout")
    return None


def _complete(path: Path) -> bool:
    summary = path / "summary.json"
    return summary.exists() and bool(json.loads(summary.read_text()).get("complete"))


def summarise_run(path: Path, annotations: Annotations | None = None) -> RunSummary:
    """Summarise the run in ``path`` (see :class:`~apex.domain.services.RunSummary`)."""
    run_dir = RunDirectory(path)
    config = composition.snapshot_config(run_dir)
    history = run_dir.history()
    summary_path = path / "summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    provenance_path = path / "provenance.json"
    provenance = json.loads(provenance_path.read_text()) if provenance_path.exists() else None
    posterior = selected = None
    posterior_metrics: dict[str, dict[str, float]] = {}
    if history:
        results = RunResults(path, composition.build_layout(config))
        selected = results.selected_iteration
        posterior = results.posterior_iteration
        record = next(r for r in history if r.iteration == posterior and r.accepted)
        posterior_metrics = record.metrics or {}
    runs = 0
    events = path / "events.jsonl"
    if events.exists():
        for line in events.read_text().splitlines():
            raw = json.loads(line)
            if raw.get("event") == "forecast_done":
                runs += int(raw["n_ok"]) + int(raw["n_failed"])
    state: Literal["complete", "stopped", "not started"] = (
        "not started" if not history else "complete" if summary.get("complete") else "stopped"
    )
    return RunSummary(
        run_id=None if provenance is None else provenance.get("run_id"),
        path=str(path),
        name=config.run.name,
        method=config.method.name,
        state=state,
        converged=summary.get("converged"),
        iterations=[r.iteration for r in history],
        accepted_iterations=[r.iteration for r in history if r.accepted],
        posterior_iteration=posterior,
        selected_iteration=selected,
        members=history[-1].n_members if history else None,
        data_mismatch={
            r.iteration: (r.data_mismatch_mean, r.data_mismatch_std) for r in history if r.accepted
        },
        metrics={r.iteration: r.metrics or {} for r in history if r.accepted},
        posterior_metrics=posterior_metrics,
        simulator_runs=runs,
        annotations=dict(annotations or {}),
        provenance=provenance,
    )


def _finite(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None


def steps_document(summary: "step_summary.RunSummary") -> RunSteps:
    """Convert a summarizer's result into its JSON document (non-finite values as ``None``)."""
    return RunSteps(
        workflow=summary.workflow,
        name=summary.name,
        path=str(summary.path),
        method=summary.method,
        status=summary.status,
        steps=[
            StepInfo(
                iteration=s.iteration,
                step=number,
                n_members=s.n_members,
                rejected_before=s.rejected_before,
                control=None if s.control is None else _finite(s.control),
                label=s.label,
            )
            for number, s in enumerate(summary.steps)
        ],
        selected_step=summary.step_of(summary.selected_iteration),
        metrics=[
            StepStatistics(
                metric=m.metric,
                label=m.label,
                iteration=m.iteration,
                mean=_finite(m.mean),
                std=_finite(m.std),
                median=_finite(m.median),
                p10=_finite(m.p10),
                p90=_finite(m.p90),
            )
            for m in summary.metrics
        ],
        selected_iteration=summary.selected_iteration,
        forecasts=summary.forecasts,
        notes=list(summary.notes),
    )


def summarise_prediction(path: Path) -> PredictionSummary:
    """Summarise a finished prediction."""
    config, result = prediction.load_prediction(path)
    labels = PERCENTILE_LABELS[config.prediction.percentile_convention]
    final: dict[str, dict[str, dict[str, float]]] = {}
    for name, forecast in result.forecasts.items():
        final[name] = {}
        for v, vector in enumerate(forecast.vectors):
            values = forecast.values[v, -1, :]
            stats = {"mean": float(values.mean()), "std": float(values.std(ddof=1))}
            stats.update({label: float(np.percentile(values, q)) for label, q in labels})
            final[name][vector] = stats
    some = next(iter(result.forecasts.values()))
    return PredictionSummary(
        path=str(path),
        name=config.prediction.name,
        run_dir=str(config.prediction.run_dir),
        ensembles=list(result.forecasts),
        vectors=list(some.vectors),
        times=list(some.times),
        first_prediction_time=result.first_prediction_time,
        final=final,
        scores=result.scores,
    )


def engine_plugin(name: str) -> RunEngine:
    """Return the run engine ``name``.

    Raises
    ------
    ConfigError
        If no such engine is installed (ERT's needs the ``ert`` extra).
    """
    try:
        return engines.create(name)
    except ApexError as error:
        raise ConfigError(
            f"no run engine {name!r} ({error}); the ert engine needs the `ert` extra"
        ) from error
