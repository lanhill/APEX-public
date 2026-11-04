"""``apex surrogate build|check``: surrogate models built and checked locally.

Builds run their full runs through the case's configured executor, as ``apex run`` does
(no job service yet: that is the phase 6).
"""

from pathlib import Path
from typing import Annotated

import typer

from apex.cli.progress import progress_display
from apex.core.errors import ApexError
from apex.domain.surrogate import GateResult, SurrogateBuildPlan, SurrogateBuildSummary

surrogate_app = typer.Typer(
    no_args_is_help=True,
    help="Surrogate models: build one from full runs of a case, or check one on new runs.",
)

MaxRuns = Annotated[
    int | None,
    typer.Option(
        "--max-runs", min=0, help="Run at most this many full runs now (resume continues)."
    ),
]
ShowProgress = Annotated[
    bool, typer.Option(help="Show progress on standard error (bars, or timestamped lines).")
]


def surrogate_plan_lines(plan: SurrogateBuildPlan) -> list[str]:
    """Format a build's plan (``apex validate``) for the terminal."""
    runs = "none (the run's stored forecasts)" if plan.plan == "run" else f"{plan.simulator_runs:,}"
    if plan.coarse_only:
        runs += f", {plan.coarse_only:,} of them coarse-only"
    lines = [
        f"surrogate {plan.name}: {plan.model} on a {plan.plan} design of {plan.training} "
        f"training and {plan.held_out} held-out runs",
        f"  simulator runs: {runs} ({plan.simulator}, executor {plan.executor})",
        f"  keys: {plan.keys} ({', '.join(plan.datatypes[:6])}"
        + (", ..." if len(plan.datatypes) > 6 else "")
        + f"), gated by {plan.measure.replace('_', ' ')}",
        f"  inputs: {plan.inputs} entries",
        f"  output: {plan.output_dir}",
    ]
    lines += [f"  warning: {w}" for w in plan.warnings]
    return lines


def _gate_lines(gate: GateResult) -> list[str]:
    lines = []
    for entry in gate.datatypes:
        error = "not finite" if entry.error is None else f"{entry.error:.4g}"
        verdict = "ok" if entry.passed else "FAILED"
        lines.append(
            f"  {entry.datatype}: {entry.measure.replace('_', ' ')} error {error} "
            f"(tolerance {entry.tolerance:.4g}) {verdict}"
        )
    return lines


def _summary_lines(path: Path, summary: SurrogateBuildSummary) -> list[str]:
    if summary.status != "complete":
        return [
            f"{path}: stopped with {summary.pending} run(s) to go; continue with "
            f"`apex surrogate build --resume {path}`"
        ]
    lines = [
        f"{path}: {summary.plugin} fitted on {summary.training_runs} runs, gated on "
        f"{summary.held_out_runs}"
        + (f" ({len(summary.failed)} failed)" if summary.failed else "")
        + (
            f", with {summary.coarse_only_runs} coarse-only runs"
            if summary.coarse_only_runs
            else ""
        )
    ]
    if summary.gate is not None:
        lines += _gate_lines(summary.gate)
        lines.append(f"gate {'passed' if summary.gate.passed else 'FAILED'}")
    if summary.model is not None:
        lines.append(f"model: {path / summary.model} (identity {summary.identity})")
    return lines


def register(app: typer.Typer) -> None:
    """Add ``apex surrogate`` to ``app``."""
    app.add_typer(surrogate_app, name="surrogate")


@surrogate_app.command("build")
def build_command(
    config: Annotated[Path | None, typer.Argument(help="A surrogate build (TOML).")] = None,
    resume_dir: Annotated[
        Path | None, typer.Option("--resume", help="Continue the build in this directory.")
    ] = None,
    max_runs: MaxRuns = None,
    progress: ShowProgress = True,
) -> None:
    """Build a surrogate: full runs of a design, then the fit, its domain and its gate.

    Running the same configuration again resumes it; one whose plugin, options or gate
    changed is fitted again on the runs already made. Exits 1 if the gate fails.
    """
    from apex.services import surrogates

    if (config is None) == (resume_dir is None):
        typer.secho("error: give a configuration or --resume BUILD_DIR", err=True, fg="red")
        raise typer.Exit(2)
    try:
        with progress_display(progress) as reporter:
            if resume_dir is not None:
                result = surrogates.resume_build(resume_dir, progress=reporter, max_runs=max_runs)
            else:
                assert config is not None
                result = surrogates.build(config, progress=reporter, max_runs=max_runs)
    except ApexError as error:
        typer.secho(f"error: {error}", err=True, fg=typer.colors.RED)
        raise typer.Exit(1) from error
    typer.echo("\n".join(_summary_lines(result.path, result.summary)))
    if result.summary.gate is not None and not result.summary.gate.passed:
        raise typer.Exit(1)


@surrogate_app.command("check")
def check_command(
    model: Annotated[Path, typer.Argument(help="The model's manifest (surrogate.json).")],
    against: Annotated[
        Path,
        typer.Option(
            "--against",
            help="A history-matching configuration (draws from its prior) or run directory "
            "(members of its posterior).",
        ),
    ],
    runs: Annotated[int, typer.Option("--runs", min=1, help="How many new full runs.")] = 10,
    seed: Annotated[int, typer.Option(help="Seed of the members drawn or chosen.")] = 0,
    iteration: Annotated[
        int | None, typer.Option(help="A run's iterate instead of its posterior.")
    ] = None,
    min_success_fraction: Annotated[
        float, typer.Option(min=0.0, max=1.0, help="Fail unless this share of the runs succeeds.")
    ] = 0.9,
    min_inside: Annotated[
        float,
        typer.Option(
            min=0.0,
            max=1.0,
            help="Fail unless this share of the successful runs is inside the model's domain.",
        ),
    ] = 0.8,
    max_runs: MaxRuns = None,
    progress: ShowProgress = True,
) -> None:
    """Gate an existing surrogate on new full runs; the model file never changes.

    Members equal to the model's own runs are left out (reported as trained on).

    The verdict goes to the build's checks/<id>/. Exits 1 if the check fails.
    """
    from apex.services import surrogates

    try:
        with progress_display(progress) as reporter:
            result = surrogates.check_surrogate(
                model, against, runs=runs, seed=seed, iteration=iteration, progress=reporter,
                max_runs=max_runs, min_success_fraction=min_success_fraction,
                min_inside=min_inside,
            )  # fmt: skip
    except ApexError as error:
        typer.secho(f"error: {error}", err=True, fg=typer.colors.RED)
        raise typer.Exit(1) from error
    summary = result.summary
    if summary.status != "complete":
        typer.echo(f"{result.path}: stopped; run the same check again to continue")
        return
    lines = [f"{result.path}: {summary.runs} runs from {summary.against} ({summary.source})"]
    if summary.trained_on:
        lines.append(f"  {len(summary.trained_on)} member(s) left out: the model's own runs")
    lines.append(
        f"  {summary.succeeded} of {summary.runs} runs succeeded, {summary.inside} inside the "
        f"model's domain (gated), {len(summary.outside)} outside"
    )
    assert summary.gate is not None
    lines += _gate_lines(summary.gate)
    lines += [f"  {note}" for note in summary.notes if note not in summary.gate.failures()]
    lines.append(f"check {'passed' if summary.passed else 'FAILED'}")
    typer.echo("\n".join(lines))
    if not summary.passed:
        raise typer.Exit(1)
