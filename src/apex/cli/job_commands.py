"""CLI commands for jobs: ``apex submit``, ``apex jobs`` and ``apex code``.

``apex submit`` puts one configuration (a history-matching run, a prediction or an
optimization) or a resume in the job service's queue, through the same calls as the REST
API and studies: the catalogue, the policy, the pools, the monitor and the notification
when the job ends. It pins the job to a frozen copy of APEX by default (``[jobs]
cli_code``): a git worktree at one commit with its own environment, so editing the checkout
never reaches the job.
"""

import json
import os
import sys
import tomllib
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Literal

import typer

from apex.core.errors import ApexError, ConfigError
from apex.domain.services import JobHandle
from apex.services.api import ApexService
from apex.services.approval import ApprovalRequired
from apex.services.jobs import FINAL

Kind = Literal["run", "prediction", "optimization"]

#: Overrides ``[jobs] cli_code`` (the test suite runs live code: no copies of the checkout).
CLI_CODE_ENV = "APEX_CLI_CODE"

Root = Annotated[
    Path | None, typer.Option(help="Services root (default: $APEX_HOME, else ~/.apex/service).")
]

code_app = typer.Typer(no_args_is_help=True, help="Frozen copies of APEX that pinned jobs run.")


def fail(error: ApexError) -> typer.Exit:
    """Print ``error`` in red and return the exit (status 1) to raise; other commands use it."""
    typer.secho(f"error: {error}", err=True, fg=typer.colors.RED)
    return typer.Exit(1)


@code_app.command("freeze")
def code_freeze(
    commit: Annotated[str, typer.Argument(help="A commit, branch or tag.")] = "HEAD",
    root: Root = None,
) -> None:
    """Make (or complete) the frozen copy of a commit; print its folder."""
    service = ApexService(root)
    try:
        uncommitted = service.code.uncommitted() if commit == "HEAD" else []
        pin = service.code.ensure(commit)
    except ApexError as error:
        raise fail(error) from error
    if uncommitted:
        typer.echo(
            f"note: {len(uncommitted)} uncommitted changes to its code are not in the copy",
            err=True,
        )
    typer.echo(str(service.code.path(pin.commit)))


@code_app.command("list")
def code_list(root: Root = None) -> None:
    """List the frozen copies, and what uses each."""
    service = ApexService(root)
    jobs = service.jobs.all_records()
    try:
        copies = service.code.copies()
    except ApexError as error:
        raise fail(error) from error
    if not copies:
        typer.echo(f"no frozen copies in {service.code.directory}")
    for copy in copies:
        users = service.code.in_use(copy.commit, jobs)
        state = f"in use: {', '.join(users)}" if users else "idle"
        typer.echo(f"{copy.commit}  {copy.subject[:70]}  [{state}]")


@code_app.command("remove")
def code_remove(
    commit: Annotated[str, typer.Argument(help="The copy's 12-character commit.")],
    root: Root = None,
) -> None:
    """Remove a frozen copy (refused while a job or a process uses it)."""
    service = ApexService(root)
    try:
        service.code.remove(commit, service.jobs.all_records)
    except ApexError as error:
        raise fail(error) from error
    typer.echo(f"removed {service.code.path(commit)}")


jobs_app = typer.Typer(no_args_is_help=True, help="The job service's jobs.")


def detect_kind(config: Path) -> Kind:
    """Return what a configuration file describes, from its sections.

    Raises
    ------
    ConfigError
        If it cannot be read, or describes none or several of the three.
    """
    try:
        sections = set(tomllib.loads(config.read_text(encoding="utf-8")))
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise ConfigError(f"cannot read {config}: {error}") from error
    marks: tuple[tuple[Kind, str], ...] = (
        ("run", "method"),
        ("prediction", "prediction"),
        ("optimization", "optimization"),
    )
    found = [kind for kind, section in marks if section in sections]
    if len(found) != 1:
        what = " and ".join(found) or "none of [method], [prediction], [optimization]"
        raise ConfigError(f"{config} has {what}: give --kind run|prediction|optimization")
    return found[0]


def command_line_code(service: ApexService, code: str | None) -> str:
    """Return the code a command-line job runs: ``--code``, ``$APEX_CLI_CODE``, ``[jobs]``.

    When freezing is only the default and APEX does not run from a git checkout (an
    installed APEX), the job runs live code, with a note.
    """
    explicit = code or os.environ.get(CLI_CODE_ENV)
    chosen = str(explicit or service.jobs.settings.jobs.cli_code)
    if chosen != "live" and not explicit and service.code.repository is None:
        typer.echo(
            "note: APEX does not run from a git checkout, so the job runs live code "
            "(set [jobs] repository in service.toml to freeze it)",
            err=True,
        )
        return "live"
    return chosen


def note_uncommitted(service: ApexService, commits: Iterable[str | None]) -> None:
    """Say what a job pinned to HEAD leaves out: the checkout's uncommitted changes."""
    pinned = {c for c in commits if c}
    if not pinned:
        return
    try:
        head, uncommitted = service.code.resolve("HEAD"), service.code.uncommitted()
    except ApexError:
        return
    if head in pinned and uncommitted:
        shown = ", ".join(uncommitted[:3]) + (", ..." if len(uncommitted) > 3 else "")
        typer.echo(
            f"note: the job runs commit {head}; {len(uncommitted)} uncommitted changes are "
            f"not in it ({shown})",
            err=True,
        )


def code_label(record: dict[str, Any]) -> str:
    """Return the code a job ran: its pinned commit, ``live``, or ``reused``.

    A job that reused an identical finished run ran no code: it shows ``reused`` and, when a
    job made that run, that job's code (the tester's H18).
    """
    spec = record.get("spec") or {}
    if "reused" in spec:
        made = spec.get("reused_code")
        return f"reused ({made})" if isinstance(made, str) else "reused"
    return str((record.get("code") or {}).get("commit") or "live")


@jobs_app.command("list")
def jobs_list(
    state: Annotated[
        list[str] | None, typer.Option(help="Only jobs in this state (repeatable).")
    ] = None,
    last: Annotated[int, typer.Option(help="Only the last N jobs (0: all).")] = 20,
    root: Root = None,
) -> None:
    """List the jobs, oldest first: id, kind, state, code, where."""
    records = ApexService(root).jobs.all_records()
    if state:
        records = [r for r in records if r["state"] in state]
    for record in records[-last:] if last else records:
        code = code_label(record)
        where = record["run_dirs"][0] if record["run_dirs"] else ""
        typer.echo(f"{record['id']}  {record['kind']:<19} {record['state']:<9} {code:<12}  {where}")


@jobs_app.command("show")
def jobs_show(
    job: Annotated[str, typer.Argument(help="A job id (or its first characters).")],
    events: Annotated[int, typer.Option(help="The last N events too.")] = 10,
    root: Root = None,
) -> None:
    """Show a job's record and its last events, as JSON."""
    service = ApexService(root)
    record = service.jobs.record(_job_id(service, job))
    shown: dict[str, Any] = {**record, "spec": {k: v for k, v in record["spec"].items()
                                                  if k != "config"}}  # fmt: skip
    every = [e.model_dump() for e in service.events(record["id"])]
    shown["events"] = every[-events:] if events else []
    typer.echo(json.dumps(shown, indent=2, default=str))


@jobs_app.command("stop")
def jobs_stop(
    job: Annotated[str, typer.Argument(help="A job id (or its first characters).")],
    root: Root = None,
) -> None:
    """Stop a job: a queued one is cancelled, a running run stops at its next checkpoint."""
    service = ApexService(root)
    job_id = _job_id(service, job)
    service.stop(job_id)
    typer.echo(f"job {job_id}: {service.jobs.record(job_id)['state']}"
               + (" (stopping at the next checkpoint)" if service.jobs.record(job_id)["state"]
                  == "running" else ""))  # fmt: skip


def _job_id(service: ApexService, prefix: str) -> str:
    matches = [str(r["id"]) for r in service.jobs.all_records() if r["id"].startswith(prefix)]
    if len(matches) != 1:
        raise fail(ApexError(f"{len(matches)} jobs match {prefix!r}"))
    return matches[0]


def submit(
    config: Annotated[
        Path | None, typer.Argument(help="A run, prediction or optimization configuration.")
    ] = None,
    resume: Annotated[
        Path | None, typer.Option(help="Resume this run directory instead.")
    ] = None,
    kind: Annotated[
        str | None, typer.Option(help="run, prediction or optimization (default: from the file).")
    ] = None,
    max_steps: Annotated[
        int | None,
        typer.Option(help="Stop after this many steps (iterations of an optimization)."),
    ] = None,
    code: Annotated[
        str | None,
        typer.Option(help="live, frozen (the last commit; a resume: the run's own) or a "
                     "commit (default: \\[jobs] cli_code, frozen)."),
    ] = None,
    priority: Annotated[str, typer.Option(help="high, normal or low.")] = "normal",
    yes: Annotated[bool, typer.Option("--yes", help="Approve a large job without asking.")] = False,
    approval_token: Annotated[
        str | None, typer.Option(help="A token approved with `apex approve`.")
    ] = None,
    force_new: Annotated[
        bool, typer.Option(help="Run again even if an identical run exists.")
    ] = False,
    wait: Annotated[bool, typer.Option(help="Wait for the job to end.")] = False,
    client: Annotated[
        str, typer.Option(help="Client name, for quotas and grants (`apex grant`).")
    ] = "user",
    annotation: Annotated[
        list[str] | None,
        typer.Option(help="An annotation 'key=value' (repeatable); 'experiment=NAME' lets a "
                     "grant of --client for that experiment approve the job."),
    ] = None,
    root: Root = None,
) -> None:  # fmt: skip
    """Run a configuration, or resume a run, in the job service's background.

    Above the approval limits, a grant of --client for the job's experiment annotation
    approves it; otherwise you are asked at a terminal (--yes: approved). Without
    a terminal's yes the command prints the pending approval token and exits with status 1:
    `apex approve TOKEN`, then the same command with --approval-token TOKEN.
    """
    if (config is None) == (resume is None):
        raise typer.BadParameter("give a configuration, or --resume RUN_DIR (not both)")
    if kind is not None and kind not in ("run", "prediction", "optimization"):
        raise typer.BadParameter("--kind is run, prediction or optimization")
    annotations = parse_annotations(annotation)
    service = ApexService(root)
    # --yes approves at the gate; otherwise the gate tries a grant first, and only then is
    # a person asked (below), so that a grant covers a job without a question.
    approve = (lambda cost: True) if yes else None
    request = SubmitRequest(kind, max_steps, priority, force_new, client, annotations)

    try:
        chosen = command_line_code(service, code)
        before = {r["id"] for r in service.jobs.all_records()}
        try:
            handle = _submit(service, config, resume, request, chosen, approve, approval_token)
        except ApprovalRequired as needed:
            if not _confirmed(service, needed):
                print_pending(needed)
                raise typer.Exit(1) from None
            service.approvals.approve(needed.token, by=f"{client}: interactive")
            handle = _submit(service, config, resume, request, chosen, None, needed.token)
    except ApexError as error:
        raise fail(error) from error
    record = service.jobs.record(handle.id)
    pinned = (record.get("code") or {}).get("commit")
    note_uncommitted(service, [pinned])
    where = record["run_dirs"][0] if record["run_dirs"] else "?"
    reused = ""
    if handle.id in before:
        reused = " (already queued or running)"
    elif handle.reused:
        reused = " (an identical run: reused)"
    typer.echo(f"{handle.kind} job {handle.id}{reused}: {record['state']}, "
               f"code {code_label(record)}")  # fmt: skip
    typer.echo(f"  folder: {where}")
    if record["state"] not in FINAL:
        typer.echo(f"  follow: apex jobs show {handle.id}  |  stop: apex jobs stop {handle.id}")
    if wait:
        status = service.wait(handle.id)
        typer.echo(f"job {status.state}" + (f": {status.error}" if status.error else ""))
        if status.state != "complete":
            raise typer.Exit(1)


def parse_annotations(given: Iterable[str] | None) -> dict[str, str]:
    """Return ``key=value`` options as annotations.

    Raises
    ------
    typer.BadParameter
        If one has no ``=`` or no key.
    """
    annotations: dict[str, str] = {}
    for item in given or []:
        key, sep, value = item.partition("=")
        if not sep or not key.strip():
            raise typer.BadParameter(f"--annotation {item!r}: give key=value")
        annotations[key.strip()] = value
    return annotations


@dataclass(frozen=True)
class SubmitRequest:
    """What ``apex submit`` was asked for, besides the file and the approval."""

    kind: str | None
    max_steps: int | None
    priority: str
    force_new: bool
    client: str
    annotations: dict[str, str]


def _confirmed(service: ApexService, needed: ApprovalRequired) -> bool:
    """Ask at a terminal whether to approve a pending submission; ``False`` without one.

    Scripts and CI have no terminal: they get the pending token (release tester N7,
    2026-10-20), where a question read end-of-file and printed only "Aborted.".
    """
    if not sys.stdin.isatty():
        return False
    record = service.approvals.store.approval(needed.token) or {}
    cost = record.get("cost") or {}
    question = (
        f"This may take {cost.get('simulator_runs')} simulator runs in "
        f"{cost.get('runs')} runs. Run it?"
    )
    try:
        return typer.confirm(question)
    except (typer.Abort, EOFError):
        return False


def print_pending(needed: ApprovalRequired) -> None:
    """Print a refused submission's pending token and how to approve it (on stderr)."""
    # The reason only: the service's message says how in the API's form (approval_token=),
    # and the CLI's is below (release tester C9, 2026-10-24).
    typer.secho(f"not submitted: {needed.reason}", err=True, fg=typer.colors.YELLOW)
    typer.echo(f"pending approval token: {needed.token}", err=True)
    typer.echo(
        f"approve it with `apex approve {needed.token}`, then run this command again with "
        f"--approval-token {needed.token}",
        err=True,
    )


def _submit(
    service: ApexService,
    config: Path | None,
    resume: Path | None,
    request: SubmitRequest,
    code: str,
    approve: Any,  # noqa: ANN401 (the approval callback)
    approval_token: str | None,
) -> JobHandle:
    notes = request.annotations or None
    if resume is not None:
        return service.resume_run(
            resume, max_steps=request.max_steps, code=code, priority=request.priority,
            client=request.client,
        )  # fmt: skip
    assert config is not None
    config = config.resolve()
    what = request.kind or detect_kind(config)
    if what == "run":
        return service.submit_run(
            config, max_steps=request.max_steps, force_new=request.force_new,
            priority=request.priority, approve=approve, approval_token=approval_token,
            code=code, client=request.client, annotations=notes,
        )  # fmt: skip
    if what == "prediction":
        return service.submit_prediction(
            config, priority=request.priority, code=code, client=request.client,
            annotations=notes,
        )  # fmt: skip
    from apex.services.optimization import load_optimization_config

    # Placed as apex optimize places it (runs/<name> beside the file), not under the root.
    return service.submit_optimization(
        load_optimization_config(config), max_iterations=request.max_steps,
        priority=request.priority, approve=approve, approval_token=approval_token, code=code,
        client=request.client, annotations=notes,
    )  # fmt: skip


def register(app: typer.Typer) -> None:
    """Add the job commands to the ``apex`` application."""
    app.command("submit")(submit)
    app.add_typer(jobs_app, name="jobs")
    app.add_typer(code_app, name="code")
