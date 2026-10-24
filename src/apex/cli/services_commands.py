"""CLI commands over the application services.

``apex plugins``, ``schema``, ``validate``, ``stop``, ``approve``, ``gc``, and the groups
``apex study`` (``run``, ``status``, ``compare``), ``apex grant`` (``create``, ``list``,
``revoke``), ``apex service`` (``start``, ``stop``, ``status``, ``reindex``) and
``apex runs`` (``list``, ``show``). They translate to
:class:`ApexService` calls, as the Python client and, later, the REST API and MCP server
do. When a submission needs approval (``[approval]`` of ``service.toml``: more than
20,000 simulator runs or 20 runs by default), the CLI asks; ``--yes`` answers yes.
"""

import json
import subprocess
import tomllib
from pathlib import Path
from typing import Annotated

import typer

from apex.core.errors import ApexError
from apex.domain.services import OptimizationPlan, RunPlan
from apex.services.api import SCHEMA_KINDS, ApexService, config_schema
from apex.services.plugins import all_plugins
from apex.storage.run_directory import RunDirectory

#: Seconds ``apex service stop`` waits for the service to end before saying it still is.
STOP_WAIT_S = 30.0
#: ``apex service stop``'s exit code when the service is still stopping after the wait.
STILL_STOPPING = 2

Root = Annotated[
    Path | None, typer.Option(help="Services root (default: $APEX_HOME, else ~/.apex/service).")
]

#: ``apex gc``: a group, so installed packages can add subcommands.
gc_app = typer.Typer(help="Forget runs whose folder is gone; prune what the service keeps.")


def gc_app_commands() -> set[str]:
    """Return the names of ``apex gc``'s subcommands (those installed packages added)."""
    return {g.name for g in gc_app.registered_groups if g.name} | {
        c.name for c in gc_app.registered_commands if c.name
    }


study_app = typer.Typer(no_args_is_help=True, help="Studies: cases x methods x variants x seeds.")
service_app = typer.Typer(no_args_is_help=True, help="The job service of a root.")
runs_app = typer.Typer(no_args_is_help=True, help="The run catalogue of a root.")
grant_app = typer.Typer(
    no_args_is_help=True, help="Budget grants: approve an experiment's spending ahead."
)


def _fail(error: ApexError) -> typer.Exit:
    typer.secho(f"error: {error}", err=True, fg=typer.colors.RED)
    return typer.Exit(1)


def optimization_plan_lines(plan: OptimizationPlan) -> list[str]:
    """Format an optimization's plan for the terminal."""
    bound = "unknown" if plan.max_evaluations is None else f"{plan.max_evaluations:,}"
    lines = [
        f"optimization {plan.name}: {plan.optimizer} on {plan.objective}, "
        f"{plan.n_controls} controls, {plan.n_realizations} realization(s)",
        f"  at most {bound} evaluations of the objective",
    ]
    if plan.max_simulator_runs:  # none for an analytic objective
        lines.append(f"  at most {plan.max_simulator_runs:,} simulator runs")
    lines += [f"  warning: {w}" for w in plan.warnings]
    return lines


def plan_lines(plan: RunPlan) -> list[str]:
    """Describe a run plan in a few lines (shared with the case-study runner)."""
    runs = "?" if plan.max_simulator_runs is None else f"{plan.max_simulator_runs:,}"
    hours = "" if plan.estimated_hours is None else f", about {plan.estimated_hours:.1f} h"
    data = ", ".join(f"{role} {n}" for role, n in plan.data.items())
    lines = [
        f"run {plan.run_id[:12]}: {plan.method}, {plan.members} members, {plan.n_state} unknowns",
        f"  data: {data}",
        f"  at most {plan.max_forecasts} forecasts, {runs} simulator runs ({plan.simulator}"
        + (f", {plan.licences} licences" if plan.licences else "")
        + (f", {plan.parallel_runs} at once" if plan.parallel_runs else "")
        + f"){hours}",
    ]
    if plan.checkpoint_mb is not None and plan.forecast_memory_mb is not None:
        lines.append(
            f"  about {plan.checkpoint_mb:,.1f} MB per checkpointed iterate, "
            f"{plan.forecast_memory_mb:,.1f} MB of memory per forecast"
        )
    lines += [f"  warning: {w}" for w in plan.warnings]
    return lines


def register(app: typer.Typer) -> None:
    """Add the services commands to the ``apex`` application."""

    @app.command()
    def plugins(
        as_json: Annotated[
            bool, typer.Option("--json", help="Full descriptions, as JSON.")
        ] = False,
    ) -> None:
        r"""List every plugin with its options (for clients: --json).

        A tag shows a plugin's status where it has one: \[parity] (checked against PET's
        output), \[validated] (against an independent reference), \[stable],
        \[experimental]. A plugin without a tag has no status of its own.
        """
        infos = all_plugins()  # no service: a listing reads no service.toml
        if as_json:
            typer.echo(json.dumps([i.model_dump(mode="json") for i in infos], indent=2))
            return
        for info in infos:
            options = ", ".join(info.options.get("properties", {}))
            status = f" [{info.status}]" if info.status else ""
            if info.unavailable:
                status += f" [unavailable: {info.unavailable}]"
            typer.echo(f"{info.registry:<18} {info.name:<18}{status} {info.summary}")
            if options:
                typer.echo(f"{'':<37}options: {options}")

    @app.command()
    def schema(
        kind: Annotated[
            str, typer.Argument(help=f"One of {', '.join(SCHEMA_KINDS)}.")
        ] = "history_match",
    ) -> None:
        """Print the JSON Schema of a configuration kind (what its TOML file accepts)."""
        try:
            typer.echo(json.dumps(config_schema(kind), indent=2))  # no service needed
        except ApexError as error:
            raise _fail(error) from error

    @app.command()
    def validate(
        config: Annotated[
            Path, typer.Argument(help="A run, prediction, optimization or study (TOML).")
        ],
        as_json: Annotated[bool, typer.Option("--json", help="The validation, as JSON.")] = False,
        root: Root = None,
    ) -> None:
        """Validate a configuration and plan its cost; nothing is simulated."""
        from apex.services.prediction import load_prediction_config

        service = ApexService(root)
        try:
            document = tomllib.loads(config.read_text())
        except (OSError, tomllib.TOMLDecodeError) as error:
            raise _fail(ApexError(f"cannot read {config}: {error}")) from error
        kind = next(iter(document), "")
        if "optimization" in document and "optimizer" in document:
            kind = "optimization"  # as the REST API tells them
        try:
            if kind == "optimization":
                result = service.validate_optimization(config)
            elif kind == "study":
                result = service.validate_study(config)
            elif kind == "prediction":
                load_prediction_config(config)
                typer.echo(f"{config}: a valid prediction configuration")
                return
            else:
                result = service.validate(config, check_output=True)  # as `apex submit` would
        except ApexError as error:
            raise _fail(error) from error
        if as_json:
            typer.echo(result.model_dump_json(indent=2))
        else:
            for plan in [result.plan] if result.plan else result.plans:
                typer.echo("\n".join(plan_lines(plan)))
            if result.optimization is not None:
                typer.echo("\n".join(optimization_plan_lines(result.optimization)))
            for problem in result.errors:
                typer.secho(f"error: {problem}", err=True, fg=typer.colors.RED)
            for note in result.notes:
                typer.echo(note)
            total = [p.max_simulator_runs for p in result.plans]
            if len(total) > 1 and all(t is not None for t in total):
                most = sum(t or 0 for t in total)
                typer.echo(f"study: {len(total)} runs, at most {most:,} simulator runs")
        if not result.valid:
            raise typer.Exit(1)

    @app.command()
    def stop(
        run_dir: Annotated[Path, typer.Argument(help="Directory of a running run.")],
    ) -> None:
        """Ask a running run to stop at its next checkpoint (resume continues it)."""
        RunDirectory(run_dir).request_stop()
        typer.echo(f"{run_dir}: will stop at its next checkpoint")

    @app.command()
    def approve(
        token: Annotated[str, typer.Argument(help="The pending approval's token.")],
        root: Root = None,
    ) -> None:
        """Approve a pending submission (then submit it again with its token)."""
        service = ApexService(root)
        try:
            record = service.approvals.approve(token, by="user: apex approve")
        except ApexError as error:
            raise _fail(error) from error
        cost = record["cost"]
        typer.echo(
            f"approved {token}: {cost['simulator_runs']} simulator runs in {cost['runs']} runs"
        )

    @gc_app.callback(invoke_without_command=True)
    def gc(
        ctx: typer.Context,
        root: Root = None,
        chats: Annotated[
            bool,
            typer.Option(help="Also apply the chat history's limits now (\\[chat] "
                         "retention_days and max_history_mb)."),
        ] = False,
        guests: Annotated[
            bool,
            typer.Option(help="Also apply the guests' limits now (\\[guests] "
                         "retention_days and max_storage_mb; guests' work only)."),
        ] = False,
        hosts: Annotated[bool, typer.Option(hidden=True)] = False,
        host: Annotated[list[str] | None, typer.Option(hidden=True)] = None,
        static_days: Annotated[float | None, typer.Option(hidden=True)] = None,
        deleted_run: Annotated[list[Path] | None, typer.Option(hidden=True)] = None,
        dry_run: Annotated[
            bool, typer.Option(help="With --code: list what would go, remove nothing.")
        ] = False,
        code: Annotated[
            bool,
            typer.Option(help="Also remove frozen copies of APEX no job or process uses that "
                         "are older than \\[jobs] keep_code_days, and run the "
                         "installed maintenance tasks."),
        ] = False,
    ) -> None:  # fmt: skip
        """Forget catalogue entries whose run directory is gone (never deletes runs).

        The options also prune chats, guests' work and frozen copies; installed packages
        add subcommands (e.g. the hosts' clean-up).
        """
        if ctx.invoked_subcommand is not None:
            if root or chats or guests or code or dry_run:
                raise typer.BadParameter(
                    f"apex gc's options go without a subcommand: apex gc "
                    f"{ctx.invoked_subcommand} --help lists the subcommand's own"
                )
            return
        if hosts or host or static_days is not None or deleted_run:
            # Kept for one release: the hosts' clean-up is a subcommand now, of
            # the package that runs members on hosts, when it is installed.
            moved = "apex gc --hosts (with --host, --static-days, --deleted-run)"
            new = "apex gc hosts" if "hosts" in gc_app_commands() else "a subcommand"
            typer.secho(f"{moved} is now {new}: see apex gc --help", err=True,
                        fg=typer.colors.YELLOW)  # fmt: skip
            raise typer.Exit(2)
        if dry_run and not code:
            raise typer.BadParameter("--dry-run goes with --code")
        service = ApexService(root)
        gone = service.jobs.store.forget_missing()
        typer.echo(f"forgot {len(gone)} runs whose directory is gone")
        for entry in gone:
            typer.echo(f"  {entry.run_id[:12]}  {entry.path}")
        if chats:
            from apex.services.daemon import prune_chats

            deleted = prune_chats(service.root, service.jobs.settings)
            typer.echo(f"deleted {len(deleted)} chats (retention and size limits)")
        if guests:
            from apex.services.guest_retention import prune_guests

            pruned = prune_guests(service)
            typer.echo(f"deleted {len(pruned)} guest submissions ([guests] limits)")
            for unit in pruned:
                typer.echo(f"  {unit.guest}  job {unit.job}  {unit.size / 1e6:.1f} MB")
        if code:
            keep = service.jobs.settings.jobs.keep_code_days
            try:
                removed = service.code.collect(service.jobs.all_records, keep, dry_run=dry_run)
            except ApexError as error:
                raise _fail(error) from error
            verb = "would remove" if dry_run else "removed"
            typer.echo(f"{verb} {len(removed)} frozen copies unused for {keep:g} days")
            for copy in removed:
                typer.echo(f"  {copy.commit}  {copy.subject[:70]}")
            from apex.services.daemon import maintenance_context, run_maintenance

            context = maintenance_context(service.root, service.jobs.settings)
            for name, lines in run_maintenance(context, dry_run=dry_run).items():
                for line in lines:
                    typer.echo(f"{name}: {line}")

    app.add_typer(gc_app, name="gc")

    app.add_typer(study_app, name="study")
    app.add_typer(grant_app, name="grant")
    app.add_typer(service_app, name="service")
    app.add_typer(runs_app, name="runs")


@study_app.command("run")
def study_run(
    study: Annotated[Path, typer.Argument(help="Study specification (TOML).")],
    yes: Annotated[
        bool, typer.Option("--yes", help="Approve a large study without asking.")
    ] = False,
    wait: Annotated[bool, typer.Option(help="Wait for the study to finish.")] = False,
    max_concurrent_jobs: Annotated[
        int | None, typer.Option(help="Jobs at once (default: service.toml, else 2).")
    ] = None,
    priority: Annotated[str, typer.Option(help="high, normal or low.")] = "normal",
    approval_token: Annotated[
        str | None, typer.Option(help="A token approved with `apex approve`.")
    ] = None,
    code: Annotated[
        str | None,
        typer.Option(help="live, frozen (the last commit, for every cell) or a commit "
                     "(default: \\[jobs] cli_code, frozen)."),
    ] = None,
    root: Root = None,
) -> None:  # fmt: skip
    """Validate and submit a study; its runs go on in the background."""
    from apex.cli.job_commands import command_line_code, note_uncommitted, print_pending
    from apex.services.approval import ApprovalRequired

    service = ApexService(root, max_concurrent_jobs=max_concurrent_jobs)

    def approve(plan: object) -> bool:
        total = getattr(plan, "max_simulator_runs", None)
        runs = len(getattr(plan, "cells", []))
        question = f"The study may take {total} simulator runs in {runs} runs. Run it?"
        return yes or typer.confirm(question)

    try:
        handle = service.submit_study(
            study, approve=approve, approval_token=approval_token, priority=priority,
            code=command_line_code(service, code),
        )  # fmt: skip
    except ApprovalRequired as needed:  # the CLI's --approval-token, as apex submit's
        print_pending(needed)
        raise typer.Exit(code=1) from needed
    except ApexError as error:
        raise _fail(error) from error
    cells = [service.jobs.record(c) for c in service.jobs.record(handle.id)["children"]]
    note_uncommitted(service, [(c.get("code") or {}).get("commit") for c in cells])
    shared = len(handle.run_ids) - len(set(handle.run_ids))
    note = f" ({shared} identical, run once)" if shared else ""
    typer.echo(f"study job {handle.id}: {len(handle.run_ids)} runs submitted{note}")
    typer.echo(f"follow it with: apex study status {_name(study)}")
    if wait:
        status = service.wait(handle.id)
        typer.echo(f"study {status.state}")


def _name(study: Path) -> str:
    return str(tomllib.loads(study.read_text())["study"]["name"])


@study_app.command("status")
def study_status(
    name: Annotated[str, typer.Argument(help="The study's name.")],
    root: Root = None,
) -> None:
    """Show a study's state and every run's."""
    service = ApexService(root)
    try:
        status = service.study_status(name)
    except ApexError as error:
        raise _fail(error) from error
    typer.echo(f"study {name}: {status.state}")
    counts: dict[str, int] = {}
    for state in status.children.values():
        counts[state] = counts.get(state, 0) + 1
    typer.echo("  " + ", ".join(f"{n} {s}" for s, n in sorted(counts.items())))


@study_app.command("compare")
def study_compare(
    name: Annotated[str, typer.Argument(help="The study's name.")],
    metric: Annotated[list[str] | None, typer.Option(help="'metric:label'; repeatable.")] = None,
    as_json: Annotated[bool, typer.Option("--json", help="The comparison, as JSON.")] = False,
    root: Root = None,
) -> None:
    """Compare a study's variants at each run's posterior."""
    service = ApexService(root)
    try:
        comparison = service.compare(name, metrics=metric)
    except ApexError as error:
        raise _fail(error) from error
    if as_json:
        typer.echo(comparison.model_dump_json(indent=2))
        return
    for item in comparison.metrics:
        typer.echo(f"\n{item.case}, {item.metric}")
        for v in item.variants:
            spread = "" if v.std is None else f" ± {v.std:.4g}"
            typer.echo(f"  {v.label():<32} n={v.n}  {v.mean:.5g}{spread}")
        for d in item.differences:
            test = "" if d.p_value is None else f", Wilcoxon p = {d.p_value:.3g}"
            se = "" if d.standard_error is None else f" ± {d.standard_error:.3g}"
            typer.echo(f"  {d.label()} - {d.reference_label()}: {d.mean_difference:+.4g}{se}{test}")
    for note in comparison.notes:
        typer.secho(f"note: {note}", fg=typer.colors.YELLOW)
    if comparison.missing:
        typer.echo(f"not finished yet: {', '.join(comparison.missing)}")
    for cell in comparison.skipped:
        typer.echo(f"not run: {cell['label']}: {cell['reason']}")


@grant_app.command("create")
def grant_create(
    client: Annotated[str, typer.Argument(help="The client the grant is for (e.g. planner).")],
    experiment: Annotated[str, typer.Argument(help="The submissions' `experiment` annotation.")],
    max_simulator_runs: Annotated[
        int, typer.Option("--max-simulator-runs", help="The total budget of simulator runs.")
    ],
    expires: Annotated[
        str, typer.Option("--expires", help="How long it lasts: 7d, 12h or 30m (at most 30d).")
    ],
    root: Root = None,
) -> None:
    """Approve up to a budget of one client's experiment ahead, until it expires."""
    from apex.services.approval import parse_duration

    service = ApexService(root)
    try:
        record = service.grants.create(
            client,
            experiment,
            max_simulator_runs,
            parse_duration(expires),
            by="user: apex grant create",
        )
    except ApexError as error:
        raise _fail(error) from error
    typer.echo(
        f"grant {record['id']}: client {client!r}, experiment {experiment!r}, "
        f"{max_simulator_runs:,} simulator runs, until {record['expires']}"
    )


@grant_app.command("list")
def grant_list(
    show_all: Annotated[
        bool, typer.Option("--all", help="Also the expired and revoked grants.")
    ] = False,
    root: Root = None,
) -> None:
    """List the active grants: id, client, experiment, used/budget, expiry."""
    from apex.services.approval import grant_state

    records = ApexService(root).grants.records(include_inactive=show_all)
    if not records:
        typer.echo("no grants" if show_all else "no active grants")
    for record in records:
        used = f"{record['used_simulator_runs']:,}/{record['max_simulator_runs']:,}"
        state = f"  {grant_state(record)}" if show_all else ""
        typer.echo(
            f"{record['id']}  {record['client']:<12} {record['experiment']:<20} {used:>15}"
            f"  {record['expires']}{state}"
        )


@grant_app.command("revoke")
def grant_revoke(
    grant_id: Annotated[str, typer.Argument(metavar="ID", help="The grant's id.")],
    root: Root = None,
) -> None:
    """Revoke a grant at once (submissions already let through are not stopped)."""
    try:
        record = ApexService(root).grants.revoke(grant_id, by="user: apex grant revoke")
    except ApexError as error:
        raise _fail(error) from error
    typer.echo(
        f"revoked {grant_id}: {record['used_simulator_runs']:,} of "
        f"{record['max_simulator_runs']:,} simulator runs were used"
    )


@service_app.command("start")
def service_start(
    root: Root = None,
    http: Annotated[
        str | None,
        typer.Option(
            help="Also serve the REST API at HOST:PORT (the api extra, installed by "
            "default). "
            "Keep 127.0.0.1 unless a TLS proxy is in front."
        ),
    ] = None,
) -> None:
    """Start the job service of a root in the background (one per root)."""
    import sys
    import time

    from apex.services.api import default_root
    from apex.services.jobs import service_pid

    home = root or default_root()
    home.mkdir(parents=True, exist_ok=True)
    pid = service_pid(home)
    if pid is not None:
        typer.echo(f"the service of {home} already runs (pid {pid})")
        return
    address = _free_address(http) if http is not None else None
    # By module name: the api layer is a sibling of cli, which must not import it.
    command = [sys.executable, "-m", "apex.cli.service_daemon", str(home)]
    if http is not None:
        command = [sys.executable, "-m", "apex.api.server", str(home), "--http", http]
    with (home / "service.log").open("a") as log:
        process = subprocess.Popen(
            command,
            stdout=log,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )
    # It writes its pid once started: a few seconds, more on a loaded machine (its imports
    # and agents took ~30 s while a test suite ran; it waited 10 s, gave up and left the
    # service running unattended, 2026-10-06).
    deadline = time.monotonic() + _start_timeout()
    while time.monotonic() < deadline:
        pid = service_pid(home)
        # With --http, started only once its own server answers: the pid comes before the
        # port is bound, and a port taken meanwhile ends the service (the release tester).
        serving = address is None or _answers(*address)
        if pid is not None and serving and process.poll() is None:
            typer.echo(f"service of {home} started (pid {pid}); log: {home / 'service.log'}")
            if http is not None:
                typer.echo(f"REST API: http://{http}/v1 (token: apex service token)")
            return
        if process.poll() is not None:
            break  # it exited: its log says why
        time.sleep(0.1)
    pid = service_pid(home)
    if pid is not None and process.poll() is None and address is not None:
        # Ours runs, but its REST API did not answer in time: say so, not "started".
        raise _fail(ApexError(
            f"the service of {home} runs (pid {pid}), but its REST API did not answer at "
            f"http://{http}/v1 in time; see {home / 'service.log'} (stop it with apex "
            "service stop)"
        ))  # fmt: skip
    if pid is not None:  # another start won the race: that one runs
        typer.echo(f"service of {home} already running (pid {pid})")
        return
    _stop_unready(process)
    raise _fail(ApexError(f"the service did not start; see {home / 'service.log'}"))


#: How long ``service start`` waits for the service to write its pid (seconds; overridden by
#: ``$APEX_SERVICE_START_TIMEOUT``).
START_TIMEOUT_S = 60.0


def _start_timeout() -> float:
    import os

    try:
        value = float(os.environ.get("APEX_SERVICE_START_TIMEOUT", START_TIMEOUT_S))
    except ValueError:
        return START_TIMEOUT_S
    return value if value > 0 else START_TIMEOUT_S  # 0 would stop a healthy service at once


def _free_address(http: str) -> tuple[str, int]:
    """Return ``--http``'s host and port, refusing a port another program already uses.

    Raises
    ------
    typer.Exit
        If the address is malformed or its port is taken.
    """
    import socket

    host, _, port_text = http.rpartition(":")
    host = host or "127.0.0.1"
    try:
        port = int(port_text)
    except ValueError as error:
        raise _fail(ApexError(f"--http takes HOST:PORT or PORT, not {http!r}")) from error
    if not 0 < port < 65536:
        raise _fail(ApexError(f"--http needs a port from 1 to 65535, not {port}"))
    with socket.socket(socket.AF_INET6 if ":" in host else socket.AF_INET) as probe:
        # As the server binds (uvicorn sets it): a port in TIME_WAIT after a stop is free,
        # one another program listens on is still refused.
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            probe.bind((host, port))
        except OSError as error:
            raise _fail(ApexError(
                f"cannot serve on {host}:{port}: {error.strerror or error}. Another program "
                "may use that port: choose another (--http HOST:PORT)"
            )) from error  # fmt: skip
    return host, port


def _answers(host: str, port: int) -> bool:
    """Whether APEX's REST API answers at ``host:port`` (its ``/v1/health``)."""
    import json as _json
    import urllib.error
    import urllib.request

    url = f"http://{'[' + host + ']' if ':' in host else host}:{port}/v1/health"
    # Straight to the port: an http_proxy of the environment would answer instead.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(url, timeout=1.0) as response:
            return bool(_json.loads(response.read()).get("status") == "ok")
    except (OSError, ValueError, urllib.error.URLError):
        return False


def _stop_unready(process: "subprocess.Popen[bytes]") -> None:
    """Stop a service that did not start in time, with its process group (no stray service)."""
    import os
    import signal

    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=10)
    except (ProcessLookupError, PermissionError):
        return
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            return


@service_app.command("token")
def service_token(
    root: Root = None,
    rotate: Annotated[
        bool, typer.Option(help="Replace the token (old clients are refused).")
    ] = False,
    ui: Annotated[
        bool,
        typer.Option("--ui", help="A guest's UI token instead: they see only their own work, "
                     "within <root>/ui_access.toml."),
    ] = False,
    guest: Annotated[
        str, typer.Option(help="The guest (with --ui); each guest has a token of their own.")
    ] = "guest",
    revoke: Annotated[
        bool, typer.Option(help="Delete the guest's token (with --ui); refused at once.")
    ] = False,
    guests: Annotated[
        bool, typer.Option("--list-guests", help="List the guests who have a token, and when "
                           "each expires.")
    ] = False,
    owner: Annotated[
        bool, typer.Option("--owner", help="The owner's web-UI token (with --ui): it "
                           "expires, unlike the API token, which the web UI refuses.")
    ] = False,
) -> None:  # fmt: skip
    r"""Print the REST API's bearer token (created on first use, readable by you only).

    Web-UI tokens (--ui: a guest's, or with --owner the owner's) expire \[ui] token_days after
    they were issued; --rotate issues a new one. The API token does not expire.
    """
    from apex.services.access import ACCESS_FILE
    from apex.services.api import ApexService, default_root
    from apex.services.auth import (
        OWNER_UI_FILE,
        api_token,
        expires_at,
        guest_token_files,
        owner_ui_token,
        revoke_guest,
        ui_token,
    )

    base = root or default_root()
    days = ApexService(base).jobs.settings.ui.token_days
    if guests:
        for name, path in guest_token_files(base).items():
            typer.echo(f"{name}\texpires {expires_at(path, days).isoformat(timespec='seconds')}")
        return
    if not ui:
        typer.echo(api_token(base, rotate=rotate))
        return
    if owner:
        typer.echo(owner_ui_token(base, create=True, rotate=rotate))
        when = expires_at(base / OWNER_UI_FILE, days).isoformat(timespec="seconds")
        typer.secho(f"expires {when}", err=True)
        return
    try:
        if revoke:
            gone = revoke_guest(base, guest)
            typer.echo(f"revoked {guest}'s token" if gone else f"{guest} has no token")
            return
        typer.echo(ui_token(base, create=True, rotate=rotate, guest=guest))
    except ValueError as error:
        typer.secho(f"error: {error}", err=True, fg=typer.colors.RED)
        raise typer.Exit(1) from error
    if not (base / ACCESS_FILE).exists():
        typer.secho(
            f"note: {base / ACCESS_FILE} does not exist, so the guest can open no folder and "
            "use no plugin yet (see docs/ui.md, 'Who sees what')",
            err=True, fg=typer.colors.YELLOW,
        )  # fmt: skip


@service_app.command("stop")
def service_stop(
    root: Root = None,
    timeout: Annotated[
        float | None,
        typer.Option(
            help="Seconds to wait for it to end (default 30); exit code 2 if it "
            "is still stopping then."
        ),
    ] = None,
) -> None:
    """Stop the job service (running jobs go on; their workers record their end)."""
    import os
    import signal
    import time

    from apex.services.api import default_root
    from apex.services.jobs import service_pid

    home = root or default_root()
    pid = service_pid(home)
    if pid is None:
        typer.echo(f"no service runs for {home}")
        return
    os.kill(pid, signal.SIGTERM)
    # The service ends its HTTP connections (GRACEFUL_SHUTDOWN_S) and the scheduler's
    # current pass first, which can take a while (a monitor's model call): wait, then say
    # it is still stopping rather than that it did not stop.
    wait = STOP_WAIT_S if timeout is None else timeout
    deadline = time.monotonic() + wait
    while time.monotonic() < deadline:
        if service_pid(home) is None:
            typer.echo(f"service of {home} stopped")
            return
        time.sleep(0.2)
    # Not an error: exit code 2 tells a restart script to wait longer (or try again).
    typer.secho(
        f"the service (pid {pid}) is still stopping after {wait:g} s: it ends its current "
        "work first; `apex service status` shows when it has", err=True,
    )  # fmt: skip
    raise typer.Exit(STILL_STOPPING)


@service_app.command("status")
def service_status(root: Root = None) -> None:
    """Show whether the service runs, and the jobs by state."""
    from apex.services.jobs import service_pid

    service = ApexService(root)
    pid = service_pid(service.root)
    state = f"running (pid {pid})" if pid else "not running (embedded mode)"
    typer.echo(f"service: {state}")
    counts: dict[str, int] = {}
    for record in service.jobs.all_records():
        counts[record["state"]] = counts.get(record["state"], 0) + 1
    typer.echo("jobs: " + (", ".join(f"{n} {s}" for s, n in sorted(counts.items())) or "none"))


@service_app.command("reindex")
def service_reindex(root: Root = None) -> None:
    """Rebuild the run catalogue from the run directories."""
    from apex.services.maintenance import reindex

    service = ApexService(root)
    entries = reindex(service.root, service.jobs.store)
    typer.echo(f"indexed {len(entries)} runs")


@runs_app.command("list")
def runs_list(
    annotation: Annotated[
        list[str] | None, typer.Option(help="Only runs with this annotation, 'key=value'.")
    ] = None,
    root: Root = None,
) -> None:
    """List the catalogued runs (oldest first)."""
    wanted = dict(a.split("=", 1) for a in annotation or [])
    for entry in ApexService(root).jobs.store.entries():
        if all(entry.annotations.get(k) == v for k, v in wanted.items()):
            typer.echo(f"{entry.run_id[:12]}  {entry.created}  {entry.name:<32} {entry.path}")


@runs_app.command("show")
def runs_show(
    run: Annotated[str, typer.Argument(help="A run id (or its first characters).")],
    root: Root = None,
) -> None:
    """Show a run's summary, as JSON."""
    service = ApexService(root)
    matches = [e for e in service.jobs.store.entries() if e.run_id.startswith(run)]
    if len(matches) != 1:
        raise _fail(ApexError(f"{len(matches)} runs match {run!r}"))
    typer.echo(service.run_summary(matches[0].run_id).model_dump_json(indent=2))


tools_app = typer.Typer(no_args_is_help=True, help="The tools LLM clients use.")

ClientName = Annotated[str, typer.Option(help="Client name, for quotas and the call log.")]
ModelId = Annotated[str | None, typer.Option(help="The model driving the calls, for the log.")]


@tools_app.command("list")
def tools_list() -> None:
    """List the tools, marking those that spend compute (they need an approval token)."""
    from apex.services.tools import TOOLS

    for tool in TOOLS.values():
        mark = " [spends compute]" if tool.spends_compute else ""
        typer.echo(f"{tool.name:<20}{mark} {tool.description}")


@tools_app.command("export")
def tools_export(
    format_: Annotated[
        str, typer.Option("--format", help="openai, anthropic or json-schema.")
    ] = "openai",
) -> None:
    """Print the tools as function-calling definitions (e.g. for Ollama or vLLM)."""
    from apex.services.tools import export

    if format_ not in ("openai", "anthropic", "json-schema"):
        raise _fail(ApexError(f"unknown format {format_!r}; use openai, anthropic, json-schema"))
    typer.echo(json.dumps(export(format_), indent=2))  # type: ignore[arg-type]


@tools_app.command("call")
def tools_call(
    name: Annotated[str, typer.Argument(help="The tool.")],
    arguments: Annotated[str, typer.Option("--json", help="Its arguments, as JSON.")] = "{}",
    client: ClientName = "llm",
    model: ModelId = None,
    root: Root = None,
) -> None:
    """Run one tool call (e.g. from an agent loop over a local model); print its result."""
    from apex.services.tools import ToolSession, call

    try:
        parsed = json.loads(arguments)
    except json.JSONDecodeError as error:
        raise _fail(ApexError(f"--json is not JSON: {error}")) from error
    session = ToolSession(ApexService(root), client=client, model=model, host="cli")
    typer.echo(json.dumps(call(name, parsed, session), indent=2, default=str))


def register_tools(app: typer.Typer) -> None:
    """Add ``apex tools`` and ``apex mcp``."""
    app.add_typer(tools_app, name="tools")

    @app.command()
    def mcp(
        http: Annotated[
            str | None,
            typer.Option(
                help="Serve Streamable HTTP at HOST:PORT, behind the API token of "
                "`apex service token` (default: stdio)."
            ),
        ] = None,
        client: ClientName = "mcp",
        model: ModelId = None,
        budget_runs: Annotated[
            int | None, typer.Option(help="Simulator runs this session may submit.")
        ] = None,
        root: Root = None,
    ) -> None:
        """Serve APEX's tools to an MCP host (Claude Code, ...); mcp extra, installed by default."""
        from apex.cli.mcp_server import serve_http, serve_stdio
        from apex.services.tools import ToolSession

        session = ToolSession(
            ApexService(root),
            client=client,
            model=model,
            host="mcp-http" if http else "mcp-stdio",
            budget_runs=budget_runs,
        )
        if http:
            host, _, port = http.rpartition(":")
            host = host or "127.0.0.1"
            option = f" --root {root}" if root else ""
            # On stderr: stdout is the protocol's over stdio, and a log's over HTTP.
            typer.secho(
                f"APEX's MCP server at http://{host}:{port}/mcp. Every request needs "
                f"'Authorization: Bearer <token>'; the token is printed by "
                f"`apex service token{option}` (requests without it get 401).",
                err=True,
            )
            serve_http(session, host, int(port))
        else:
            serve_stdio(session)
