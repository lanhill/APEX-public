"""Write a QA/QC report (``apex qaqc``): Markdown, and one coverage figure per data type."""

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from apex.reporting.plots import FONT_SIZE, INK, MUTED, PRIOR, _pyplot
from apex.reporting.result_plots import _save


class _Movement(Protocol):
    @property
    def prior_std(self) -> float: ...
    @property
    def std(self) -> float: ...
    @property
    def beyond(self) -> tuple[float, float, float]: ...


class _Coverage(Protocol):
    @property
    def datatype(self) -> str: ...
    @property
    def times(self) -> tuple[str, ...]: ...
    @property
    def forecast(self) -> NDArray[np.float64]: ...
    @property
    def observations(self) -> NDArray[np.float64]: ...
    @property
    def below(self) -> NDArray[np.bool_]: ...
    @property
    def above(self) -> NDArray[np.bool_]: ...

    @property
    def outside(self) -> int: ...


class _Report(Protocol):
    @property
    def run(self) -> str: ...
    @property
    def iteration(self) -> int: ...
    @property
    def coverage(self) -> Sequence[_Coverage]: ...
    @property
    def data_scores(self) -> Sequence[tuple[str, float]]: ...
    @property
    def type_scores(self) -> Sequence[tuple[str, float]]: ...
    @property
    def pair_scores(self) -> Sequence[tuple[str, float]]: ...
    @property
    def gains_max(self) -> Sequence[tuple[str, float]]: ...
    @property
    def gains_mean(self) -> Sequence[tuple[str, float]]: ...
    @property
    def movement(self) -> Mapping[str, _Movement]: ...
    @property
    def notes(self) -> Sequence[str]: ...


def qaqc_markdown(report: _Report) -> str:
    """Return the report as Markdown."""
    lines = [f"# QA/QC: {report.run}, iteration {report.iteration}", ""]
    lines += [
        "## Coverage",
        "",
        "Observations outside the forecast range (the forecast misses them).",
        "",
    ]
    lines += ["| data type | outside | of |", "| --- | --- | --- |"]
    lines += [f"| {c.datatype} | {c.outside} | {len(c.times)} |" for c in report.coverage]
    sections = (
        ("Mahalanobis: data", "Largest scores (Oliver 2020): data the model cannot reach.",
         report.data_scores),
        ("Mahalanobis: data types", "Each type's series on its leading component.",
         report.type_scores),
        ("Mahalanobis: pairs", "Largest joint scores of two data.", report.pair_scores),
        ("Kalman gain, largest entry", "Which data move which parameters most.",
         report.gains_max),
        ("Kalman gain, mean", "The mean gain on the parameter group.", report.gains_mean),
    )  # fmt: skip
    for title, text, rows in sections:
        if not rows:
            continue
        lines += ["", f"## {title}", "", text, "", "| | score |", "| --- | --- |"]
        lines += [f"| {label} | {value:.4g} |" for label, value in rows]
    lines += ["", "## Movement from the prior", ""]
    lines += ["| parameters | prior std | std | > 1 std | > 2 std | > 3 std |"]
    lines += ["| --- | --- | --- | --- | --- | --- |"]
    for name, m in report.movement.items():
        beyond = " | ".join(f"{b:.1f}%" for b in m.beyond)
        lines.append(f"| {name} | {m.prior_std:.4g} | {m.std:.4g} | {beyond} |")
    lines += [f"- note: {note}" for note in report.notes]
    return "\n".join(lines) + "\n"


def coverage_figure(coverage: _Coverage, path: Path, *, font_size: float = FONT_SIZE) -> Path:
    """Plot one data type's forecast members and observations; missed ones marked ``x``."""
    plt = _pyplot(font_size)
    fig, ax = plt.subplots(figsize=(7.0, 4.2), layout="constrained")
    x = np.arange(len(coverage.times))
    ax.plot(x, coverage.forecast, color=MUTED, linewidth=0.6, alpha=0.6)
    inside = ~(coverage.below | coverage.above)
    ax.plot(x[inside], coverage.observations[inside], linestyle="none", marker="o",
            markersize=5, color=INK, label="observed")  # fmt: skip
    if (~inside).any():
        ax.plot(x[~inside], coverage.observations[~inside], linestyle="none", marker="x",
                markersize=8, color=PRIOR, label="outside the forecast")  # fmt: skip
    ticks = x[:: max(1, len(x) // 8)]
    ax.set_xticks(ticks, [coverage.times[i] for i in ticks], rotation=30, ha="right")
    ax.set_ylabel(coverage.datatype)
    ax.set_title(f"{coverage.datatype}: {coverage.outside} of {len(x)} outside", loc="left")
    ax.legend(loc="best", frameon=False, fontsize=font_size * 0.75)
    png = _save(fig, path)
    plt.close(fig)
    return png


def write_qaqc(report: _Report, output: Path, *, font_size: float = FONT_SIZE) -> list[Path]:
    """Write ``qaqc.md`` and the coverage figures (PNG and PDF) into ``output``."""
    output.mkdir(parents=True, exist_ok=True)
    markdown = output / "qaqc.md"
    markdown.write_text(qaqc_markdown(report))
    written = [markdown]
    for c in report.coverage:
        name = c.datatype.replace(":", "_").replace("/", "_")
        written.append(coverage_figure(c, output / f"coverage_{name}", font_size=font_size))
    return written
