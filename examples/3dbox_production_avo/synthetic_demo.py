"""Synthetic 3dBox study with production and AVO data: four history matches compared.

The 3dBox case (60x60x5 cells, three injectors and three producers, OPM Flow) with the
log-permeability and the porosity of every cell uncertain (36,000 unknowns, 100 members),
matched to the wells' rates and to RMS maps of two AVO angle stacks at two vintages:

1. Check the case: the observations and maps ``prepare.py`` wrote from the true model, and
   the four configurations (``apex validate``'s plan).
2. History-match with ES-MDA (4 steps, alpha = 4) and GIES/RLM-MAC (at most 10
   iterations), each with and without adaptive localization (``adaptive_prior`` with the
   fixed threshold 3/sqrt(N)). Results go to ``results/<experiment>/``.
3. Tabulate the prior and final member means of the metrics: normalized data mismatch
   (all data and each AVO map), RMSE and SSIM of ln PERMX and porosity against the
   true model, and the spread kept (``results/comparison/final_metrics.csv``); and the
   same at every kept iteration of each experiment
   (``results/comparison/metrics_by_iteration_<experiment>.csv``).
4. Plot the results, one figure per file in PNG and PDF (``--font-size``, default 12): the
   metrics over the iterations for the four experiments (``results/comparison``); per
   experiment the metrics' distributions, the observed and simulated rates with the true
   model's, and per layer the true model and the initial and final ensemble means and
   spreads of both variables (``results/<experiment>/report``).

Run it (needs OPM Flow on PATH and the ``plots`` and ``reservoir`` extras)::

    uv run python examples/3dbox_production_avo/synthetic_demo.py              # everything
    uv run python examples/3dbox_production_avo/synthetic_demo.py --plots-only
    uv run python examples/3dbox_production_avo/synthetic_demo.py --only esmda gies

The four runs make about 3,200 simulations of 100 members each forecast, a few hours on
10 cores. The notebook ``synthetic_demo.ipynb`` runs the same steps one by one.
"""

import argparse
import csv
import shutil
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from apex.domain.config import HistoryMatchConfig
from apex.reporting.result_plots import metric_comparison_figure
from apex.services import composition, validation
from apex.storage.run_results import RunResults

HERE = Path(__file__).resolve().parent
RESULTS_DIR = HERE / "results"
FONT_SIZE = 12.0
#: Experiment name -> (label for figures and tables, configuration file).
EXPERIMENTS = {
    "esmda": ("ES-MDA", "config_esmda.toml"),
    "gies": ("GIES", "config_gies.toml"),
    "esmda_noloc": ("ES-MDA, no localization", "config_esmda_noloc.toml"),
    "gies_noloc": ("GIES, no localization", "config_gies_noloc.toml"),
}
#: The state variables, as the metrics label them.
VARIABLES = ("permx", "poro")
#: The data types besides the rates (each scored on its own).
FIELD_TYPES = ("avo_10", "avo_20")
#: The label the metrics give each field type: the maps are wavelet-compressed, so they are
#: scored on their kept coefficients.
FIELD_LABELS = {datatype: f"wavelet({datatype})" for datatype in FIELD_TYPES}


# ------------------------------------------------------------------------------- 1. case


def load_config(name: str, output_dir: Path | None = None) -> HistoryMatchConfig:
    """Return an experiment's configuration, its run written to ``results/<name>``."""
    config = composition.load_config(HERE / EXPERIMENTS[name][1])
    run = config.run.model_copy(update={"output_dir": output_dir or RESULTS_DIR / name})
    return config.model_copy(update={"run": run})


def plan(name: str) -> str:
    """Return the plan of an experiment, as ``apex validate`` gives it (nothing is run)."""
    config = load_config(name)
    p = validation.plan(config, composition.build_run(config), None)
    return (
        f"{p.method}, {p.members} members, {p.n_state} unknowns, "
        f"{sum(p.data.values())} data; at most {p.max_forecasts} forecasts, "
        f"{p.max_simulator_runs} simulations"
    )


def true_response(name: str = "esmda") -> NDArray[np.float64]:
    """Return the true model's noise-free data at the observations' keys, in their order.

    The rates from ``true_rates.csv`` and the maps from ``true_response.npz``, both written
    by ``prepare.py`` from the truth's run. They are assembled at the raw keys and mapped by
    the run's data transforms (the maps' wavelet compression), so that they line up with
    the run's data, as the report expects.
    """
    with (HERE / "true_rates.csv").open() as handle:
        rates = {(r["time"], r["datatype"]): float(r["value"]) for r in csv.DictReader(handle)}
    maps = np.load(HERE / "true_response.npz")
    workflow = composition.build_run(load_config(name)).workflow
    keys = workflow.raw_observations.keys
    values = [
        maps[f"{key.datatype}@{key.time}"][key.index]
        if key.index is not None
        else rates[(str(key.time), key.datatype)]
        for key in keys
    ]
    return workflow.transform(np.asarray(values, dtype=np.float64)[:, None])[:, 0]


# --------------------------------------------------------------------------- 2. matching


def run_experiments(
    names: Sequence[str] = tuple(EXPERIMENTS), *, rerun: bool = False
) -> dict[str, Path]:
    """Run each experiment into ``results/<name>``; existing results are kept unless ``rerun``."""
    runs = {}
    for name in names:
        run_dir = RESULTS_DIR / name
        if rerun and run_dir.exists():
            shutil.rmtree(run_dir)
        if not (run_dir / "summary.json").exists():
            print(f"running {name} ...", flush=True)
            composition.start_config(load_config(name), None)
        runs[name] = run_dir
    return runs


# ---------------------------------------------------------------------------- 3. results


def load_results(runs: dict[str, Path]) -> dict[str, RunResults]:
    """Load the saved per-iteration results of each experiment."""
    return {name: composition.load_results(path) for name, path in runs.items()}


def summary_row(name: str, run: RunResults) -> dict[str, object]:
    """Prior and final member means of the metrics, the spread kept and the cost."""
    final = run.posterior_iteration

    def mean(metric: str, label: str, iteration: int) -> float | str:
        try:
            iterations, means = run.metric_means(metric, label)
        except KeyError:  # a run that did not record the metric
            return ""
        return means[iterations.index(iteration)]

    prior, posterior = run.iterate(0).state, run.iterate(final).state
    row: dict[str, object] = {
        "experiment": EXPERIMENTS[name][0],
        "forecasts": len(run.history),  # including the prior and rejected trials
        "final_iteration": final,
        "members": posterior[VARIABLES[0]].shape[-1],
    }
    for column, label in (("all", "all"), *FIELD_LABELS.items()):
        row[f"normalized_data_mismatch_{column}_prior"] = mean("normalized_data_mismatch", label, 0)
        row[f"normalized_data_mismatch_{column}_final"] = mean(
            "normalized_data_mismatch", label, final
        )
    for variable in VARIABLES:
        for metric in ("rmse", f"ssim_{variable}"):
            column = metric if metric != "rmse" else f"rmse_{variable}"
            row[f"{column}_prior"] = mean(metric, variable, 0)
            row[f"{column}_final"] = mean(metric, variable, final)
        spread = posterior[variable].std(axis=-1) / prior[variable].std(axis=-1)
        row[f"spread_{variable}"] = float(np.mean(spread))
    return row


def write_table(results: dict[str, RunResults], name: str = "comparison") -> Path:
    """Write the experiments' summary rows to ``results/<name>/final_metrics.csv``."""
    out = RESULTS_DIR / name
    out.mkdir(parents=True, exist_ok=True)
    table = out / "final_metrics.csv"
    rows = [summary_row(n, run) for n, run in results.items()]
    with table.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(
            {k: round(v, 4) if isinstance(v, float) else v for k, v in row.items()} for row in rows
        )
    return table


def iteration_rows(run: RunResults) -> list[dict[str, object]]:
    """Member means of the metrics at every kept iteration, and the spread kept there.

    One row per kept iteration (the prior is 0): normalized data mismatch (all data and
    each AVO map), RMSE and SSIM of each variable against the true model, and the spread,
    the iterate's standard deviation over the prior's, averaged over the cells.
    """
    columns = {"mismatch all": ("normalized_data_mismatch", "all")}
    columns |= {f"mismatch {t}": ("normalized_data_mismatch", FIELD_LABELS[t]) for t in FIELD_TYPES}
    for variable in VARIABLES:
        columns[f"RMSE {variable}"] = ("rmse", variable)
        columns[f"SSIM {variable}"] = (f"ssim_{variable}", variable)
    means = {}
    for column, (metric, label) in columns.items():
        try:
            means[column] = dict(zip(*run.metric_means(metric, label), strict=True))
        except KeyError:  # a run that did not record the metric
            means[column] = {}
    prior = run.iterate(0).state
    rows: list[dict[str, object]] = []
    for iteration in run.accepted_iterations:
        row: dict[str, object] = {"iteration": iteration}
        row |= {column: values.get(iteration, "") for column, values in means.items()}
        state = run.iterate(iteration).state
        for variable in VARIABLES:
            spread = state[variable].std(axis=-1) / prior[variable].std(axis=-1)
            row[f"spread {variable}"] = float(np.mean(spread))
        rows.append(row)
    return rows


def write_iteration_tables(
    results: dict[str, RunResults], name: str = "comparison"
) -> dict[str, list[dict[str, object]]]:
    """Write ``results/<name>/metrics_by_iteration_<experiment>.csv`` for each experiment.

    Returns each experiment's rows (``iteration_rows``), so that callers display them
    without loading every iterate again; the CSV rounds them to 4 decimals, as ``write_table``.
    """
    out = RESULTS_DIR / name
    out.mkdir(parents=True, exist_ok=True)
    tables = {}
    for experiment, run in results.items():
        rows = tables[experiment] = iteration_rows(run)
        with (out / f"metrics_by_iteration_{experiment}.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(
                {k: round(v, 4) if isinstance(v, float) else v for k, v in row.items()}
                for row in rows
            )
    return tables


def markdown_table(rows: Sequence[dict[str, object]]) -> str:
    """Return ``rows`` as a Markdown table, numbers to four significant digits."""

    def cell(value: object) -> str:
        return f"{value:.4g}" if isinstance(value, float) else str(value)

    header = list(rows[0])
    lines = ["| " + " | ".join(header) + " |", "|" + "---:|" * len(header)]
    lines += ["| " + " | ".join(cell(row[c]) for c in header) + " |" for row in rows]
    return "\n".join(lines)


# ---------------------------------------------------------------------------- 4. figures


def plot_comparison(
    results: dict[str, RunResults], font_size: float = FONT_SIZE, name: str = "comparison"
) -> list[Path]:
    """One figure per metric: the experiments' member means over the iterations."""
    out = RESULTS_DIR / name
    out.mkdir(parents=True, exist_ok=True)
    panels = [
        ("normalized_data_mismatch", "all", "Normalized data mismatch, all data", True),
        *[
            ("normalized_data_mismatch", t, f"Normalized data mismatch, {t}", True)
            for t in FIELD_TYPES
        ],
        ("rmse", "permx", "RMSE of ln PERMX against the true model", False),
        ("rmse", "poro", "RMSE of porosity against the true model", False),
        ("ssim_permx", "permx", "SSIM of ln PERMX with the true model", False),
        ("ssim_poro", "poro", "SSIM of porosity with the true model", False),
    ]
    figures = []
    for metric, label, title, log in panels:
        series = {}
        for n, r in results.items():
            try:
                series[EXPERIMENTS[n][0]] = r.metric_means(metric, label)
            except KeyError:  # a run that did not record the metric
                continue
        if not series:
            continue
        figures.append(
            metric_comparison_figure(
                series,
                out / f"{metric}_{label}",
                title=title,
                ylabel=f"{metric} (member mean)",
                log=log,
                font_size=font_size,
            )
        )
    return figures


def plot_experiment(
    run_dir: Path, truth: NDArray[np.float64] | None, font_size: float = FONT_SIZE
) -> list[Path]:
    """Write one experiment's figures (``results/<name>/report``): metrics, data, fields."""
    return composition.write_report_figures(run_dir, truth_response=truth, font_size=font_size)


def main(argv: Sequence[str] | None = None) -> None:
    """Run the whole study (or only redraw the figures)."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--plots-only", action="store_true", help="reuse the saved results")
    parser.add_argument("--rerun", action="store_true", help="rerun experiments with results")
    parser.add_argument(
        "--only", nargs="+", choices=list(EXPERIMENTS), help="run only these experiments"
    )
    parser.add_argument(
        "--font-size", type=float, default=FONT_SIZE, help="font size of the figures (points)"
    )
    args = parser.parse_args(argv)
    names = args.only or list(EXPERIMENTS)

    for name in names:
        print(f"{EXPERIMENTS[name][0]}: {plan(name)}", flush=True)
    if args.plots_only:
        runs = {n: RESULTS_DIR / n for n in names}
    else:
        runs = run_experiments(names, rerun=args.rerun)
    results = load_results(runs)
    truth = true_response()
    written = plot_comparison(results, font_size=args.font_size)
    for run_dir in runs.values():
        written += plot_experiment(run_dir, truth, font_size=args.font_size)
    table = write_table(results)
    tables = write_iteration_tables(results)
    print(f"wrote {len(written)} figures (PNG + PDF) and tables under {RESULTS_DIR}/")
    print(table.read_text())
    for name, rows in tables.items():
        print(f"\n{EXPERIMENTS[name][0]}, at each kept iteration:")
        print(markdown_table(rows))


if __name__ == "__main__":
    main()
