"""Entry point for the ``apex`` command."""

import json
from collections.abc import Callable
from pathlib import Path
from typing import Annotated

import typer

from apex import __version__
from apex.cli.agent_commands import agent_app
from apex.cli.agent_commands import register as register_agent
from apex.cli.ert_commands import register as register_ert
from apex.cli.extensions import install as install_extensions
from apex.cli.job_commands import code_app
from apex.cli.job_commands import register as register_jobs
from apex.cli.progress import progress_display
from apex.cli.services_commands import gc_app, register_tools
from apex.cli.services_commands import register as register_services
from apex.core.errors import ApexError, RunStopped
from apex.domain.config import HistoryMatchConfig
from apex.domain.results import HistoryMatchResult
from apex.reporting.summary import history_table, metrics_table
from apex.services import composition
from apex.storage.run_directory import RunDirectory

app = typer.Typer(no_args_is_help=True, help="APEX: Agentic Platform for Ensemble eXploration.")

MaxSteps = Annotated[
    int | None,
    typer.Option(help="Stop after this many analysis steps; continue later with `apex resume`."),
]
ShowProgress = Annotated[
    bool,
    typer.Option(
        help="Show progress on standard error: bars on a terminal, timestamped lines otherwise."
    ),
]


@app.callback()
def main() -> None:
    """APEX command-line interface."""


@app.command()
def version() -> None:
    """Print the installed APEX version."""
    typer.echo(__version__)


@app.command()
def run(
    config: Annotated[Path, typer.Argument(help="Run configuration (TOML).")],
    max_steps: MaxSteps = None,
    progress: ShowProgress = True,
) -> None:
    """Start a history-matching run."""
    with progress_display(progress) as reporter:
        result = _guard(lambda: composition.start(config, max_steps, reporter))
    _report(result, _control(lambda: composition.load_config(config)))


@app.command()
def resume(
    run_dir: Annotated[Path, typer.Argument(help="Directory of the run to continue.")],
    max_steps: MaxSteps = None,
    progress: ShowProgress = True,
) -> None:
    """Continue a stopped or interrupted run from its last checkpoint."""
    with progress_display(progress) as reporter:
        result = _guard(lambda: composition.resume(run_dir, max_steps, reporter))
    _report(result, _control(lambda: composition.snapshot_config(RunDirectory(run_dir))))


@app.command()
def predict(
    config: Annotated[Path, typer.Argument(help="Prediction configuration (TOML).")],
    progress: ShowProgress = True,
) -> None:
    """Forecast ensembles of a history-matching run beyond history.

    Starts the prediction, or resumes it: members already forecast are not run again.
    """
    from apex.reporting.prediction_report import final_time_table, scores_table
    from apex.services import prediction

    with progress_display(progress) as reporter:
        try:
            result = prediction.predict(config, reporter)
        except RunStopped as stopped:
            typer.echo(f"stopped: {stopped}")
            raise typer.Exit(0) from stopped
        except ApexError as error:
            typer.secho(f"error: {error}", err=True, fg=typer.colors.RED)
            raise typer.Exit(1) from error
    convention = prediction.load_prediction_config(config).prediction.percentile_convention
    typer.echo(final_time_table(result, convention))
    table = scores_table(result.scores)
    if table is not None:
        typer.echo("\n" + table)


@app.command()
def optimize(
    config: Annotated[
        Path | None, typer.Argument(help="Optimization configuration (TOML).")
    ] = None,
    resume_dir: Annotated[
        Path | None,
        typer.Option("--resume", help="Continue the optimization run in this directory."),
    ] = None,
    max_iterations: Annotated[
        int | None,
        typer.Option(min=1, help="Stop after this many accepted iterations (resumable)."),
    ] = None,
    progress: ShowProgress = True,
) -> None:
    """Minimise an objective with one of PET's optimizers (EnOpt, GenOpt, SmcOpt, ...)."""
    from apex.services import optimization

    if (config is None) == (resume_dir is None):
        typer.secho("error: give a configuration or --resume RUN_DIR", err=True, fg="red")
        raise typer.Exit(2)
    try:
        with progress_display(progress) as reporter:
            if resume_dir is not None:
                run = optimization.resume(resume_dir, max_iterations, reporter)
            else:
                assert config is not None
                run = optimization.optimize(config, max_iterations, reporter)
    except ApexError as error:
        typer.secho(f"error: {error}", err=True, fg=typer.colors.RED)
        raise typer.Exit(1) from error
    result = run.result
    state = "finished" if run.finished else "stopped (resume with --resume)"
    typer.echo(f"{run.run_dir}: {state}; {result.message}")
    from apex.services import optimization as optimizations
    from apex.storage import optimization_directory as layout

    summary = optimizations.optimization_summary(run.run_dir)
    outer = (summary.result or {}).get("epf_iterations")
    # With an external penalty the iterations restart at each outer update: count those.
    span = (f"{outer} penalty updates" if summary.penalised and outer
            else f"{result.nit} iterations")  # fmt: skip
    typer.echo(
        f"objective {float(result.fun.mean()):.6g} after {span} "
        f"({optimizations.accepted_steps(summary)} steps accepted), "
        f"{optimizations.objective_evaluations(summary)} objective evaluations"
    )
    typer.echo("x = " + ", ".join(f"{v:.6g}" for v in result.x))
    if summary.pareto:
        # Several objectives: the line above is one point of a front.
        typer.echo(
            f"that is the front's representative, one of its {len(summary.pareto.get('x', []))} "
            "points; "
            f"the whole front: {run.run_dir / layout.PARETO}"
        )


@app.command()
def report(
    run_dir: Annotated[Path, typer.Argument(help="Directory of a run or of a prediction.")],
    plots: Annotated[
        bool, typer.Option(help="Also write the result figures (PNG and PDF).")
    ] = False,
    output: Annotated[
        Path | None, typer.Option(help="Directory for the figures (default: RUN_DIR/report).")
    ] = None,
    font_size: Annotated[float, typer.Option(help="Font size of the figures (points).")] = 12.0,
) -> None:
    """Print the data mismatch history and performance metrics of a run; optionally write figures.

    For a prediction directory (``apex predict``), print its forecast at the last time
    and its scores; optionally write its figures.
    """
    from apex.storage.prediction_directory import PredictionDirectory

    if PredictionDirectory.holds_prediction(run_dir):
        _report_prediction(run_dir, plots=plots, output=output, font_size=font_size)
        return
    from apex.services import optimization as optimizations

    if optimizations.holds_optimization(run_dir):
        _report_optimization(run_dir, plots=plots, output=output, font_size=font_size)
        return
    if run_dir.is_dir() and not (run_dir / "config.json").is_file():
        # A job's folder exists from its submission; its configuration only once it starts.
        # Without this, an optimization's folder fell through to a history match's report.
        message = (f"error: {run_dir} holds no run yet: it has no configuration snapshot "
                   "(config.json), so its job has not started, or it is not a run, "
                   "prediction or optimization directory")  # fmt: skip
        typer.secho(message, err=True, fg=typer.colors.RED)
        raise typer.Exit(1)
    history = RunDirectory(run_dir).history()
    control = _control(lambda: composition.snapshot_config(RunDirectory(run_dir)))
    typer.echo(history_table(history, control))
    table = metrics_table(history, _selected_iteration(run_dir))
    if table is not None:
        typer.echo("\n" + table)
    if plots:
        for figure in _guard_paths(
            lambda: composition.write_report_figures(run_dir, output, font_size=font_size)
        ):
            typer.echo(f"wrote {figure}")


def _report_optimization(
    run_dir: Path, *, plots: bool, output: Path | None, font_size: float
) -> None:
    """Print an optimization's state and result; with ``plots``, write its figures."""
    from apex.services import optimization as optimizations
    from apex.storage import optimization_directory as layout

    summary = optimizations.optimization_summary(run_dir)
    iterations = None if summary.penalised else (summary.result or {}).get("iterations")
    simulated = f", {summary.evaluations} simulated" if summary.simulated else ""
    typer.echo(
        f"{summary.optimizer} on {summary.objective}: {summary.state}, "
        f"{optimizations.accepted_steps(summary)} steps accepted after the start"
        + (f" in {iterations} iterations" if iterations else "")
        + f", {optimizations.objective_evaluations(summary)} objective evaluations{simulated}"
    )
    if summary.result is not None:
        result = summary.result
        if "objectives" in result:
            pairs = zip(result["objective_names"], result["objectives"], strict=True)
            typer.echo("representative: " + ", ".join(f"{n} {v:.6g}" for n, v in pairs))
        else:
            typer.echo(f"objective {float(result['objective']):.6g}")
        typer.echo(f"stopped: {result.get('message', '')}")
    front = layout.read_pareto(run_dir)
    if front is not None:
        typer.echo(f"front: {len(front['x'])} members, hypervolume {front['hypervolume']:.6g}")
    if plots:
        for figure in _guard_paths(
            lambda: optimizations.write_optimization_figures(run_dir, output, font_size=font_size)
        ):
            typer.echo(f"wrote {figure}")


@app.command()
def summary(
    paths: Annotated[
        list[Path], typer.Argument(help="Run directories, or folders holding runs (searched).")
    ],
    output: Annotated[
        Path | None,
        typer.Option(help="Write the Markdown here (and the CSV next to it) instead of stdout."),
    ] = None,
    title: Annotated[str, typer.Option(help="Title of the report.")] = "Results summary",
) -> None:
    """Summarise runs: every metric at every step (iterates, forecasts, optimization steps).

    Finds the runs at or under PATHS. The Markdown has an overview and a per-step table per
    run (member mean ± std of the headline metrics); the CSV, written with ``--output``,
    has every metric, label and step (mean, std, median, P10, P90).
    """
    from apex.reporting.run_summary import summary_csv, summary_markdown
    from apex.services.summaries import summarize

    found = _guard_summaries(lambda: summarize(paths))
    root = paths[0] if len(paths) == 1 and paths[0].is_dir() else None
    text = summary_markdown(found, title=title, root=root)
    if output is None:
        typer.echo(text, nl=False)
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text)
    table = output.with_suffix(".csv")
    table.write_text(summary_csv(found, root=root))
    typer.echo(f"wrote {output}\nwrote {table}")


@app.command()
def qaqc(
    run_dir: Annotated[Path, typer.Argument(help="Directory of a history-matching run.")],
    iteration: Annotated[
        int | None, typer.Option(help="Iterate to check (default: the posterior).")
    ] = None,
    output: Annotated[
        Path | None, typer.Option(help="Directory for the report (default: RUN_DIR/qaqc).")
    ] = None,
    damping: Annotated[float, typer.Option(help="lambda of the Kalman gains (0: ES).")] = 0.0,
    font_size: Annotated[float, typer.Option(help="Font size of the figures (points).")] = 12.0,
) -> None:
    """QA/QC of an iterate: coverage, Mahalanobis scores, Kalman gains, movement (PET).

    Writes qaqc.md and one coverage figure per data type (PNG and PDF).
    """
    from apex.reporting.qaqc_report import write_qaqc
    from apex.services.qaqc import qaqc_report

    report = _guard_summaries(lambda: qaqc_report(run_dir, iteration, damping=damping))
    for path in write_qaqc(report, output or run_dir / "qaqc", font_size=font_size):
        typer.echo(f"wrote {path}")


def _guard_summaries[T](action: Callable[[], T]) -> T:
    try:
        return action()
    except ApexError as error:
        typer.secho(f"error: {error}", err=True, fg=typer.colors.RED)
        raise typer.Exit(1) from error


@app.command("data-split")
def data_split(
    config: Annotated[Path, typer.Argument(help="Run configuration (TOML) of the case.")],
    prior_run: Annotated[
        Path | None,
        typer.Option(help="A run of the case with a prior forecast, for ensemble redundancy."),
    ] = None,
    max_cv: Annotated[int | None, typer.Option(help="Largest candidate CV set, in wells.")] = None,
    output: Annotated[
        Path | None,
        typer.Option(help="Directory for the results (default: CONFIG dir/data_split)."),
    ] = None,
    font_size: Annotated[float, typer.Option(help="Font size of the figures (points).")] = 12.0,
) -> None:
    r"""Explore how to split production data for cross validation.

    Ranks the wells by their observed correlation with the other wells (Luo & Ranazzi,
    Steps 1-4) and proposes \[validation] sections. Field data always stay in history
    matching.
    """
    try:
        report, paths = composition.explore_data_split(
            config, prior_run=prior_run, max_cv=max_cv, output_dir=output, font_size=font_size
        )
    except ApexError as error:
        typer.secho(f"error: {error}", err=True, fg=typer.colors.RED)
        raise typer.Exit(1) from error
    typer.echo("well        mean |r|  prior redundancy")
    for unit, score in report.scores.items():
        redundancy = "" if report.redundancy is None else f"{report.redundancy[unit]:.3f}"
        typer.echo(f"{unit:<10}  {score:8.3f}  {redundancy:>16}")
    for note in report.notes:
        typer.secho(f"note: {note}", fg=typer.colors.YELLOW)
    if report.field_datatypes:
        typer.echo(f"field data (history matching only): {', '.join(report.field_datatypes)}")
    best = report.candidates[0]
    typer.echo(f"\nlowest-scoring well as CV set ({best.n_cv} data, ratio {best.ratio:.3g}):")
    typer.echo(best.snippet())
    for path in paths:
        typer.echo(f"wrote {path}")


def _report_prediction(path: Path, *, plots: bool, output: Path | None, font_size: float) -> None:
    from apex.reporting.prediction_report import final_time_table, scores_table
    from apex.services import prediction

    try:
        config, result = prediction.load_prediction(path)
    except ApexError as error:
        typer.secho(f"error: {error}", err=True, fg=typer.colors.RED)
        raise typer.Exit(1) from error
    typer.echo(final_time_table(result, config.prediction.percentile_convention))
    table = scores_table(result.scores)
    if table is not None:
        typer.echo("\n" + table)
    if plots:
        for figure in _guard_paths(
            lambda: prediction.write_prediction_figures(path, output, font_size=font_size)
        ):
            typer.echo(f"wrote {figure}")


def _guard(action: Callable[[], HistoryMatchResult]) -> HistoryMatchResult:
    # Expected failures (bad config, too many failed members) get a clean message and
    # exit code 1; anything else is a bug and keeps its traceback.
    try:
        return action()
    except RunStopped as stopped:
        # A stop asked while the run waited outside its steps: as a stop at a
        # checkpoint, not a failure.
        typer.echo(f"stopped: {stopped}")
        raise typer.Exit(0) from stopped
    except ApexError as error:
        typer.secho(f"error: {error}", err=True, fg=typer.colors.RED)
        raise typer.Exit(1) from error


@app.command("pet-import")
def pet_import(
    pet_config: Annotated[Path, typer.Argument(help="PET TOML configuration.")],
    output_dir: Annotated[Path, typer.Argument(help="Directory for the APEX case.")],
    prior: Annotated[
        Path | None, typer.Option(help="Stored prior ensemble (.npz), if the config samples one.")
    ] = None,
    simulator: Annotated[str, typer.Option(help="APEX simulator for the runfile.")] = "opm_flow",
    seed: Annotated[int | None, typer.Option(help="Run seed (default: the config's).")] = None,
    truth: Annotated[
        Path | None,
        typer.Option(help="True model (.npz, one array per variable): a synthetic case."),
    ] = None,
    pet_geometry: Annotated[
        bool,
        typer.Option(
            "--pet-geometry/--intended-geometry",
            help="Prior correlation geometry: what PET samples, or what its keywords describe.",
        ),
    ] = True,
    grid_file: Annotated[
        Path | None,
        typer.Option(
            help="An EGRID of the deck, for geophysical data: its active cells fix the AVO "
            "traces and the gravity and displacement stations."
        ),
    ] = None,
    avo_angles_checked: Annotated[
        bool,
        typer.Option(
            help="Import multi-angle AVO data known to hold each angle's own traces "
            "(SimulatorWrap's do not: it gives every angle the last one's)."
        ),
    ] = False,
) -> None:
    """Convert a PET history-matching configuration into an APEX case."""
    from apex.storage.pet_config import import_pet_case

    try:
        result = import_pet_case(
            pet_config,
            output_dir,
            prior_file=prior,
            simulator=simulator,
            seed=seed,
            pet_geometry=pet_geometry,
            truth_file=truth,
            grid_file=grid_file,
            avo_angles_checked=avo_angles_checked,
        )
    except ApexError as error:
        typer.secho(f"error: {error}", err=True, fg=typer.colors.RED)
        raise typer.Exit(1) from error
    typer.echo(f"wrote {result.config}")
    if result.ignored:
        typer.echo("ignored (no effect in APEX): " + ", ".join(result.ignored))
    for note in result.notes:
        typer.secho(f"note: {note}", fg=typer.colors.YELLOW)
    typer.echo(f"run it with: apex run {result.config}")


def _guard_paths(action: Callable[[], list[Path]]) -> list[Path]:
    try:
        return action()
    except ApexError as error:
        typer.secho(f"error: {error}", err=True, fg=typer.colors.RED)
        raise typer.Exit(1) from error


def _control(config: Callable[[], HistoryMatchConfig]) -> str:
    """Return the method's name for its per-trial parameter (``damping`` when unknown)."""
    from apex.services.plugins import control_name

    try:
        method = config().method
    except ApexError:  # a snapshot older than its fields: the table still prints
        return "damping"
    return control_name(method.name, method.options)


def _report(result: HistoryMatchResult, control: str = "damping") -> None:
    typer.echo(history_table(result.history, control))
    status = "complete" if result.complete else "stopped (resume to continue)"
    if result.converged:
        status = "converged"
    if result.stopped_by_cv:
        status = "stopped by cross validation"
    if result.selected_iteration is not None:
        status += (
            f"; posterior = iteration {result.selected_iteration} "
            "(lowest CV normalized data mismatch)"
        )
    final = result.complete or result.converged or result.stopped_by_cv
    held = "in the posterior" if final else "in the last saved iterate (not yet the posterior)"
    typer.echo(f"\n{status}; {result.posterior.shape[-1]} members {held}")


def _selected_iteration(run_dir: Path) -> int | None:
    # With cross validation, the posterior is the minimum-CV iterate.
    summary = run_dir / "summary.json"
    if not summary.exists():
        return None
    validation = json.loads(summary.read_text()).get("validation")
    return None if validation is None else int(validation["selected_iteration"])


register_services(app)
register_tools(app)
register_agent(app)
register_ert(app)
register_jobs(app)
# Commands installed packages contribute: NORCE's hosts' commands, their agents'
# (under `apex agent`, §2 revision 1), ...
install_extensions({"": app, "agent": agent_app, "code": code_app, "gc": gc_app})


def cli() -> None:
    """Run the ``apex`` command, reporting expected errors in one line.

    An ``ApexError`` from any command (e.g. an invalid ``service.toml``) is printed as one
    line on stderr with exit code 1; any other exception is a bug and keeps its traceback.
    """
    try:
        app()
    except ApexError as error:
        typer.secho(f"error: {error}", err=True, fg=typer.colors.RED)
        raise SystemExit(1) from None
