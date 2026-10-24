"""Tables and figures of the data-split explorer.

Written by ``apex data-split``, one figure per file (PNG and PDF):

- ``correlation.png``: the ``|r|`` matrix of the production groups' observed series, on a
  sequential single-hue scale (0 to 1);
- ``scores.png``: the paper's Step-3 score per well, lowest (best CV candidate) first;
- ``redundancy.png``: the prior-ensemble redundancy per well, when a prior forecast was
  given.

The tables are ``groups.csv``, ``correlation.csv``, ``scores.csv`` and
``candidates.toml``, which holds ready-to-paste ``[validation]`` sections.
"""

import csv
from pathlib import Path

from apex.domain.data_split import SplitExploration
from apex.reporting.plots import FONT_SIZE, INK, POSTERIOR, TEXT_SECONDARY, _pyplot
from apex.reporting.result_plots import _save

#: Sequential single-hue colour map for correlation magnitudes (light = 0, dark = 1).
CORRELATION_CMAP = "Blues"


def write_exploration(
    report: SplitExploration, output_dir: Path, *, font_size: float = FONT_SIZE
) -> list[Path]:
    """Write the explorer's tables and figures to ``output_dir``; return their paths."""
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = [
        _groups_csv(report, output_dir / "groups.csv"),
        _correlation_csv(report, output_dir / "correlation.csv"),
        _scores_csv(report, output_dir / "scores.csv"),
        _candidates(report, output_dir / "candidates.toml"),
        correlation_figure(report, output_dir / "correlation", font_size=font_size),
        score_figure(
            report.scores,
            output_dir / "scores",
            title="Mean |r| with the other wells (paper, Step 3)",
            font_size=font_size,
        ),
    ]
    if report.redundancy is not None:
        paths.append(
            score_figure(
                report.redundancy,
                output_dir / "redundancy",
                title="Prior-ensemble redundancy with the other data",
                font_size=font_size,
            )
        )
    return paths


def correlation_figure(
    report: SplitExploration, path: Path, *, font_size: float = FONT_SIZE
) -> Path:
    """Heat map of ``|r|`` between the production groups; returns the PNG path."""
    plt = _pyplot(font_size)
    names = [g.datatype for g in report.groups]
    size = max(4.0, 0.45 * len(names) + 2.0)
    fig, ax = plt.subplots(figsize=(size + 1.0, size))
    image = ax.imshow(abs(report.correlation), cmap=CORRELATION_CMAP, vmin=0.0, vmax=1.0)
    ax.grid(False)  # grid lines would cut through the cells
    ax.set_xticks(range(len(names)), names, rotation=90)
    ax.set_yticks(range(len(names)), names)
    ax.set_title("|r| of the observed series (lag 0)", color=INK)
    bar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    bar.set_label("|r|", color=TEXT_SECONDARY)
    png = _save(fig, path)
    plt.close(fig)
    return png


def score_figure(
    scores: dict[str, float], path: Path, *, title: str, font_size: float = FONT_SIZE
) -> Path:
    """Horizontal bars of a per-well score, in the given order; returns the PNG path."""
    plt = _pyplot(font_size)
    units = list(scores)
    fig, ax = plt.subplots(figsize=(6.0, max(2.5, 0.4 * len(units) + 1.0)))
    ax.barh(range(len(units)), [scores[u] for u in units], color=POSTERIOR, height=0.6)
    ax.set_yticks(range(len(units)), units)
    ax.invert_yaxis()  # the best CV candidate on top
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("mean |r| (lower: better CV candidate)", color=TEXT_SECONDARY)
    ax.set_title(title, color=INK)
    png = _save(fig, path)
    plt.close(fig)
    return png


def _groups_csv(report: SplitExploration, path: Path) -> Path:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["datatype", "well", "keyword", "n_data", "role"])
        for g in report.groups:
            writer.writerow([g.datatype, g.unit, g.keyword, g.n_data, "candidate"])
        for datatype in report.field_datatypes:
            writer.writerow([datatype, "", "", "", "history_matching (field data)"])
    return path


def _correlation_csv(report: SplitExploration, path: Path) -> Path:
    names = [g.datatype for g in report.groups]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["", *names])
        for name, row in zip(names, report.correlation, strict=True):
            writer.writerow([name, *(f"{r:.6g}" for r in row)])
    return path


def _scores_csv(report: SplitExploration, path: Path) -> Path:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["rank", "well", "score", "prior_redundancy"])
        for rank, (unit, score) in enumerate(report.scores.items(), start=1):
            redundancy = "" if report.redundancy is None else f"{report.redundancy[unit]:.6g}"
            writer.writerow([rank, unit, f"{score:.6g}", redundancy])
    return path


def _candidates(report: SplitExploration, path: Path) -> Path:
    lines = [
        "# Candidate cross-validation sets from `apex data-split`.",
        "# Paste one [validation] section into the run configuration. The wells with the",
        "# lowest mean |r| with the other wells come first (Luo & Ranazzi, Step 3).",
        "# Field data always stay in history matching.",
    ]
    lines += [f"# note: {note}" for note in report.notes]
    for candidate in report.candidates:
        lines += [
            "",
            f"# {len(candidate.cv_units)} CV well(s): {candidate.n_cv} CV data, "
            f"{candidate.n_history_matching} history-matching data "
            f"(ratio {candidate.ratio:.3g})",
            *("# " + line for line in candidate.snippet().splitlines()),
        ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
