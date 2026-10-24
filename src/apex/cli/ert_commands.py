"""``apex ert``: APEX cases for Equinor's ERT, and ERT cases for APEX."""

import json
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Annotated

import tomli_w
import typer

from apex.core.errors import ApexError
from apex.domain.config import ALGORITHM_FIELDS

ert_app = typer.Typer(no_args_is_help=True, help="APEX cases for Equinor's ERT.")

#: ERT's experiment modes that ``apex ert run`` starts.
MODES = ("ensemble_experiment", "ensemble_smoother", "es_mda", "test_run")


@ert_app.command("export")
def export(
    config: Annotated[Path, typer.Argument(help="APEX run configuration (TOML).")],
    out_dir: Annotated[Path, typer.Argument(help="Folder for the ERT case.")],
    ert_defaults: Annotated[
        bool, typer.Option(help="Keep ERT's outlier rejection and STD_CUTOFF (default: off).")
    ] = False,
    overwrite: Annotated[bool, typer.Option(help="Replace a case already in OUT_DIR.")] = False,
) -> None:
    """Write CONFIG as a self-contained ERT case (case.ert, prior, observations, grid)."""
    from apex.integrations.ert.export import export_case

    try:
        result = export_case(config, out_dir, ert_defaults=ert_defaults, overwrite=overwrite)
    except ApexError as error:
        raise typer.BadParameter(str(error)) from error
    fields = ", ".join(f"{name} ({variable})" for name, variable in result.fields.items())
    typer.echo(f"wrote {result.case}: {result.realizations} realizations, fields {fields}")
    typer.echo(
        f"  data: {result.assimilated} assimilated, {result.held_out} held out (not assimilated)"
    )
    typer.echo(f"  run: apex ert run {out_dir} {' '.join(result.run_arguments)}")


@ert_app.command("import")
def import_(
    case_dir: Annotated[Path, typer.Argument(help="An exported case ERT has run.")],
    run_dir: Annotated[
        Path | None, typer.Argument(help="APEX run directory (default: CASE_DIR/apex_run).")
    ] = None,
) -> None:
    """Replay ERT's iterations into an APEX run directory (reports, summaries, comparisons)."""
    from apex.integrations.ert.importer import import_case

    try:
        result = import_case(case_dir, run_dir)
    except ApexError as error:
        raise typer.BadParameter(str(error)) from error
    first, last = result.history[0], result.history[-1]
    target = run_dir or case_dir / "apex_run"
    typer.echo(
        f"imported {len(result.history)} ERT iterates into {target}: data mismatch "
        f"{first.data_mismatch_mean:.4g} -> {last.data_mismatch_mean:.4g}"
    )
    if result.selected_iteration is not None:
        typer.echo(f"  minimum-CV iterate: {result.selected_iteration}")
    typer.echo(f"  report: apex report {target} [--plots]")


@ert_app.command("run")
def run(
    case_dir: Annotated[Path, typer.Argument(help="A folder written by `apex ert export`.")],
    mode: Annotated[str, typer.Option(help=f"ERT's experiment: {', '.join(MODES)}.")] = "es_mda",
    weights: Annotated[
        str | None, typer.Option(help="ES-MDA weights, e.g. 4,4,4,4 (mode es_mda).")
    ] = None,
    dry_run: Annotated[bool, typer.Option(help="Print the command only.")] = False,
) -> None:
    """Run ERT on an exported case, locally (ERT 26.x, the `ert` extra)."""
    if mode not in MODES:
        raise typer.BadParameter(f"mode must be one of {', '.join(MODES)}")
    case = case_dir / "case.ert"
    if not case.exists():
        raise typer.BadParameter(f"{case} is missing: run `apex ert export` first")
    # ERT's local queue runs its fm_dispatch.py (and APEX's step runs this Python): both
    # live in this environment's bin folder, which must be on PATH.
    bin_dir = str(Path(sys.executable).parent)
    env = {**os.environ, "PATH": os.pathsep.join([bin_dir, os.environ.get("PATH", "")])}
    env.pop("PYTHONPATH", None)
    executable = shutil.which("ert", path=env["PATH"])
    if executable is None:
        raise typer.BadParameter("`ert` is not on PATH: install apex[ert] (ERT 26.x)")
    command = [executable, mode]
    if weights is not None:
        if mode != "es_mda":
            raise typer.BadParameter("--weights is for --mode es_mda")
        command += ["--weights", weights]
    command.append(case.name)
    typer.echo(f"(cd {shlex.quote(str(case_dir))} && {shlex.join(command)})")
    if dry_run:
        return
    raise typer.Exit(subprocess.run(command, cwd=case_dir, env=env, check=False).returncode)


@ert_app.command("history-match")
def history_match(
    case: Annotated[Path, typer.Argument(help="An ERT case (.ert): fields, observations, steps.")],
    apex_toml: Annotated[
        Path | None, typer.Argument(help="APEX's choices (default: apex.toml next to CASE).")
    ] = None,
    methods: Annotated[
        str | None,
        typer.Option(help="Comma-separated \\[methods.<label>] to run (default: all)."),
    ] = None,
    max_steps: Annotated[
        int | None,
        typer.Option(
            help="Stop after this many analysis steps; run the same command again to resume."
        ),
    ] = None,
    progress: Annotated[bool, typer.Option(help="Show progress on standard error.")] = True,
    write_back: Annotated[
        bool, typer.Option(help="Add the iterates to the case's ERT storage (ERT's GUI).")
    ] = True,
) -> None:
    r"""History-match an ERT case with APEX's methods; ERT runs the forward model.

    The same as ERT's ``ert workflow APEX_HISTORY_MATCH CASE``. The run goes to
    CASE's folder/apex_runs/<name>/; a run already there is continued. With
    \[methods.<label>] in apex.toml, each method runs into apex_runs/<run>/<label>/.
    """
    from apex.cli.progress import progress_display
    from apex.integrations.ert.case import MethodsFailed, history_match_all

    only = [m.strip() for m in methods.split(",") if m.strip()] if methods else None
    failure: MethodsFailed | None = None
    with progress_display(progress) as reporter:
        try:
            results = history_match_all(
                case, apex_toml, methods=only, max_steps=max_steps, progress=reporter,
                write_back=write_back,
            )  # fmt: skip
        except MethodsFailed as error:
            failure, results = error, {}
            results.update(error.results)
        except ApexError as error:
            raise typer.BadParameter(str(error)) from error
    for label, result in results.items():
        last = result.history[-1]
        state = "complete" if result.complete else "stopped (resume: run the same command again)"
        typer.echo(
            f"{label + ': ' if label else ''}{state}: {len(result.history)} forecasts, data "
            f"mismatch {result.history[0].data_mismatch_mean:.4g} -> "
            f"{last.data_mismatch_mean:.4g}"
        )
    if failure is not None:
        for label, reason in failure.failed.items():
            typer.secho(f"{label}: failed: {reason}", err=True, fg=typer.colors.RED)
        raise typer.Exit(1)


def register(app: typer.Typer) -> None:
    """Add ``apex ert`` to the CLI."""
    app.add_typer(ert_app, name="ert")


@ert_app.command("compare")
def compare(
    config: Annotated[Path, typer.Argument(help="An APEX history-matching configuration.")],
    mode: Annotated[str, typer.Option(help="ERT's run model: es_mda or ensemble_smoother.")] = (
        "es_mda"
    ),
    weights: Annotated[
        str | None, typer.Option(help="ES-MDA weights, e.g. 4,4,4,4 (default: ERT's 4,2,1).")
    ] = None,
    apex_method: Annotated[
        str | None,
        typer.Option(help="APEX's method to compare, with its defaults (default: the case's)."),
    ] = None,
    replicates: Annotated[int, typer.Option(min=1, help="Seeds per method.")] = 3,
    name: Annotated[str | None, typer.Option(help="The study's name.")] = None,
    yes: Annotated[bool, typer.Option("--yes", help="Approve without asking.")] = False,
    wait: Annotated[bool, typer.Option(help="Wait for the study to finish.")] = False,
) -> None:
    r"""Compare APEX's method with ERT's on one case, over seeds: a study.

    Writes CONFIG's folder/<name>.study.toml (the case's own \[method], or --apex-method,
    against ERT's run model, paired by seed) and submits it as `apex study run` does;
    `apex study compare NAME` then compares them.
    """
    from apex.cli.services_commands import study_run
    from apex.services.composition import load_config

    try:
        case = load_config(config)
    except ApexError as error:
        raise typer.BadParameter(str(error)) from error
    study_name = name or f"{case.run.name}_apex_vs_ert"
    if apex_method is not None:
        apex_entry: dict[str, object] = {"name": apex_method}
    else:
        method = json.loads(case.method.model_dump_json(exclude_none=True))
        apex_entry = {k: v for k, v in method.items() if k in ALGORITHM_FIELDS}
    ert_entry: dict[str, object] = {"engine": "ert", "name": mode}
    if weights is not None:
        ert_entry["options"] = {"weights": [float(w) for w in weights.split(",") if w.strip()]}
    apex_label = f"apex_{apex_entry['name']}"
    spec = {
        "study": {
            "name": study_name,
            "replicates": replicates,
            "cases": {case.run.name: str(config.resolve())},
            "methods": {apex_label: apex_entry, f"ert_{mode}": ert_entry},
            "variants": [{"name": "base"}],
        }
    }
    path = config.resolve().parent / f"{study_name}.study.toml"
    path.write_text(f"# Written by `apex ert compare` from {config.name}.\n" + tomli_w.dumps(spec))
    typer.echo(f"wrote {path}")
    study_run(path, yes=yes, wait=wait)


@ert_app.command("convert")
def convert(
    case: Annotated[Path, typer.Argument(help="An ERT case (.ert) on FLOW or ECLIPSE100/300.")],
    out_dir: Annotated[Path, typer.Argument(help="Folder for the native APEX case.")],
    method: Annotated[str, typer.Option(help="APEX's method in the configuration.")] = "esmda",
    overwrite: Annotated[bool, typer.Option(help="Replace a case already in OUT_DIR.")] = False,
) -> None:
    """Write CASE as a native APEX case that runs without ERT.

    The DATA_FILE becomes a Mako deck (fields and run templates inlined, ERT's
    substitutions and GEN_KW transforms in Mako), the FIELDs and GEN_KW groups the state
    with ERT's priors, the summary observations APEX's CSV.
    """
    from apex.integrations.ert.convert import convert_case

    try:
        result = convert_case(case, out_dir, method=method, overwrite=overwrite)
    except ApexError as error:
        raise typer.BadParameter(str(error)) from error
    typer.echo(f"wrote {result.config} and {result.template}")
    for note in result.notes:
        typer.secho(f"note: {note}", fg=typer.colors.YELLOW)
    typer.echo(f"run it with: apex run {result.config}")
