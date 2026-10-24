"""Write run summaries (:class:`~apex.domain.summary.RunSummary`) as Markdown and CSV.

The Markdown has an overview (one row per run: the headline metrics at the first step and
at the result) and, per run, a table of the headline metrics at every step, as member
mean ± standard deviation (the sample std, ``ddof = 1``, as the API's step statistics and
the figures; stated in the Markdown, release tester, 2026-10-20). Headline labels are
``all`` where a metric has it, else every label of a metric with at most
:data:`MAX_LABELS`; everything else (per data type, per well) is in the CSV, one row per
run, metric, label and step.
"""

import csv
import io
import math
import os
from collections.abc import Sequence
from pathlib import Path

from apex.domain.summary import RunSummary

#: What a workflow's first step and its runs are called (default: history matching's).
WORDS = {"optimization": ("start", "evaluations")}
#: Metrics without an ``all`` label show every label up to this many.
MAX_LABELS = 3

CSV_COLUMNS = (
    "workflow", "run", "path", "method", "metric", "label", "iteration", "step", "result",
    "mean", "std", "median", "p10", "p90",
)  # fmt: skip


def headline(summary: RunSummary) -> list[tuple[str, str]]:
    """Return the ``(metric, label)`` columns of a run's tables (see the module docstring)."""
    columns = []
    for metric, labels in summary.labels().items():
        if "all" in labels:
            columns.append((metric, "all"))
        elif len(labels) <= MAX_LABELS:
            columns.extend((metric, label) for label in labels)
    return columns


def summary_markdown(
    summaries: Sequence[RunSummary], *, title: str = "Results summary", root: Path | None = None
) -> str:
    """Return the Markdown report of ``summaries``; paths relative to ``root`` if given."""
    lines = [f"# {title}", ""]
    lines.append(
        f"{len(summaries)} run(s). Values are member means ± standard deviations, the "
        "sample std over the members (ddof = 1; `apex report`'s data mismatch std is the "
        "methods' population std, ddof = 0, so it is slightly smaller). For history "
        "matching, step 0 is the prior and each later step an accepted iterate (rejected "
        "trials are counted, not listed); the *result* is the posterior, the last step or "
        "the one cross validation selected. For a prediction, each step is an ensemble "
        "forecast (the result: the posterior); forecast scores are single values. "
        "Per-type, per-well and per-vector values are in the CSV."
    )
    lines += ["", "## Overview", ""]
    columns: list[tuple[str, str]] = []
    for summary in summaries:
        columns.extend(c for c in headline(summary) if c not in columns)
    header = ["run", "method", "status", "steps", "forecasts", "result"]
    header += [_name(c) for c in columns]
    lines.append(_row(header))
    lines.append(_row(["---"] * len(header)))
    names = run_labels(summaries)
    for summary, name in zip(summaries, names, strict=True):
        first = summary.steps[0].iteration if summary.steps else 0
        result = summary.selected_iteration
        cells = [
            name,
            summary.method,
            summary.status,
            str(len(summary.steps) - 1),
            str(summary.forecasts),
            "-" if result is None else str(summary.step_of(result)),
        ]
        for metric, label in columns:
            before = summary.value(metric, label, first)
            after = None if result is None else summary.value(metric, label, result)
            cells.append(
                "-" if before is None and after is None
                else f"{_number(before.mean if before else None)} → "
                f"{_number(after.mean if after else None)}"
            )  # fmt: skip
        lines.append(_row(cells))
    lines.append("")
    lines.append("Cells: first step → result, member means.")
    for summary, name in zip(summaries, names, strict=True):
        lines += ["", *_run_section(summary, name, root)]
    return "\n".join(lines) + "\n"


def run_labels(summaries: Sequence[RunSummary]) -> list[str]:
    """Return a label per run: its name, or where names repeat its case and folder.

    The case is the folder holding ``runs/`` (``<case>/runs/<folder>``), else the run's
    parent folder; the folder is the run's own (two runs of one configuration share its
    name, not their folders). Labels that would still repeat are the runs' paths.
    """
    counts: dict[str, int] = {}
    for s in summaries:
        counts[s.name] = counts.get(s.name, 0) + 1
    labels = []
    for s in summaries:
        if counts[s.name] == 1:
            labels.append(s.name)
            continue
        # Absolute, so that a relative path such as ``copy_of_run`` (parent ``.``, whose
        # name is empty) still names its case rather than giving ``/copy_of_run``; not
        # resolved, so that a symlinked folder keeps its own name.
        path = Path(os.path.abspath(s.path))
        parent = path.parent
        case = parent.parent.name if parent.name == "runs" else parent.name
        labels.append(f"{case}/{path.name}" if case else path.name)
    repeated = {label for label in labels if labels.count(label) > 1}
    return [
        str(s.path) if label in repeated else label
        for s, label in zip(summaries, labels, strict=True)
    ]


def summary_csv(summaries: Sequence[RunSummary], *, root: Path | None = None) -> str:
    """Return every statistic of ``summaries`` as CSV (:data:`CSV_COLUMNS`)."""
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(CSV_COLUMNS)
    for s, name in zip(summaries, run_labels(summaries), strict=True):
        for row in s.metrics:
            writer.writerow(
                [
                    s.workflow, name, _path(s.path, root), s.method, row.metric, row.label,
                    row.iteration, s.step_of(row.iteration),
                    int(row.iteration == s.selected_iteration),
                    repr(row.mean), repr(row.std), repr(row.median), repr(row.p10),
                    repr(row.p90),
                ]
            )  # fmt: skip
    return buffer.getvalue()


def _run_section(summary: RunSummary, name: str, root: Path | None) -> list[str]:
    lines = [f"## {name}", ""]
    first, last = (summary.steps[0], summary.steps[-1]) if summary.steps else (None, None)
    start, runs = WORDS.get(summary.workflow, ("prior", "forecasts"))
    facts = [
        f"- path: `{_path(summary.path, root)}`",
        f"- method: {summary.method}; status: {summary.status}",
        f"- steps: {len(summary.steps) - 1} after the {start}, {summary.forecasts} {runs}; "
        f"result: step {summary.step_of(summary.selected_iteration)}",
    ]
    if first is not None and last is not None:
        facts.append(f"- members: {first.n_members} → {last.n_members}")
    facts += [f"- note: {note}" for note in summary.notes]
    lines += facts
    columns = headline(summary)
    header = ["step", "members", "rejected", "control", *(_name(c) for c in columns)]
    lines += ["", _row(header), _row(["---"] * len(header))]
    for number, step in enumerate(summary.steps):
        marker = " (result)" if step.iteration == summary.selected_iteration else ""
        name = step.label if step.label is not None else str(number)
        cells = [
            f"{name}{marker}",
            str(step.n_members),
            str(step.rejected_before),
            "-" if step.control is None else _number(step.control),
        ]
        for metric, label in columns:
            value = summary.value(metric, label, step.iteration)
            if value is None:
                cells.append("-")
            elif math.isnan(value.std):  # a single value (a forecast score), no spread
                cells.append(_number(value.mean))
            else:
                cells.append(f"{_number(value.mean)} ± {_number(value.std)}")
        lines.append(_row(cells))
    return lines


def _name(column: tuple[str, str]) -> str:
    metric, label = column
    return metric if label == "all" else f"{metric} {label}"


def _number(value: float | None) -> str:
    return "-" if value is None else f"{value:.4g}"


def _row(cells: Sequence[str]) -> str:
    return "| " + " | ".join(c.replace("|", "\\|") for c in cells) + " |"


def _path(path: Path, root: Path | None) -> str:
    if root is not None:
        try:
            return str(path.resolve().relative_to(root.resolve()))
        except ValueError:
            pass
    return str(path)
