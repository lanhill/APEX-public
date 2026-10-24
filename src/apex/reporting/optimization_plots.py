"""Figures of an optimization: objective history, front and hypervolume.

One figure per file, PNG and PDF, with ``font_size`` (as every APEX figure). The functions take
plain data (history rows, the front as ``pareto.json`` holds it); the services read the run.

- ``objective.png``: the objective of the accepted iterates (every optimizer; for several
  objectives, the representative's first objective);
- ``front.png`` (several objectives): two objectives as a scatter of the front's members (their
  means over realizations) with the representative marked; three or more as parallel
  coordinates, each objective scaled to its range on the front;
- ``hypervolume.png`` (several objectives): the hypervolume over the generations.
"""

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from apex.reporting.plots import FONT_SIZE, INK, POSTERIOR, PRIOR, _pyplot
from apex.reporting.result_plots import _save


def objective_figure(
    history: Sequence[Mapping[str, Any]], path: Path, *, font_size: float = FONT_SIZE
) -> Path:
    """Write the objective of each history row (accepted iterates) and return the PNG."""
    plt = _pyplot(font_size)
    rows = [r for r in history if "objective" in r]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(range(len(rows)), [float(r["objective"]) for r in rows], color=POSTERIOR,
            marker="o", markersize=3)  # fmt: skip
    ax.set_xlabel("accepted iterate")
    ax.set_ylabel("objective (minimised)")
    ax.set_title("Objective")
    return _save(fig, path)


def hypervolume_figure(
    history: Sequence[Mapping[str, Any]], path: Path, *, font_size: float = FONT_SIZE
) -> Path:
    """Write the hypervolume over the generations and return the PNG."""
    plt = _pyplot(font_size)
    rows = [r for r in history if "hypervolume" in r]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot([int(r.get("iteration", i)) for i, r in enumerate(rows)],
            [float(r["hypervolume"]) for r in rows], color=POSTERIOR)  # fmt: skip
    ax.set_xlabel("generation")
    ax.set_ylabel("hypervolume")
    ax.set_title("Hypervolume of the front")
    return _save(fig, path)


def front_figure(front: Mapping[str, Any], path: Path, *, font_size: float = FONT_SIZE) -> Path:
    """Write the front (two objectives: scatter; more: parallel coordinates); return the PNG."""
    plt = _pyplot(font_size)
    names = list(front["objective_names"])
    F = np.asarray(front["objectives"], dtype=float)  # (m, n_obj) as stored
    rep = int(front["representative"])
    if F.size == 0:
        raise ValueError("the front is empty")
    if len(names) == 2:
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(F[:, 0], F[:, 1], s=18, color=POSTERIOR, label="front")
        ax.scatter(F[rep, 0], F[rep, 1], s=70, facecolors="none", edgecolors=PRIOR,
                   linewidths=2, label="representative")  # fmt: skip
        ax.set_xlabel(names[0])
        ax.set_ylabel(names[1])
        ax.set_title("Pareto front (means over realizations)")
        ax.legend(frameon=False)
        return _save(fig, path)
    lo, hi = F.min(axis=0), F.max(axis=0)
    scaled = (F - lo) / np.where(hi > lo, hi - lo, 1.0)
    fig, ax = plt.subplots(figsize=(1.6 * len(names) + 2, 4.5))
    positions = np.arange(len(names))
    for i in range(F.shape[0]):
        if i != rep:
            ax.plot(positions, scaled[i], color=POSTERIOR, alpha=0.35, linewidth=1)
    ax.plot(positions, scaled[rep], color=PRIOR, linewidth=2.5, label="representative")
    ax.set_xticks(positions, names)
    ax.set_ylabel("scaled to the front's range")
    ax.set_title("Pareto front (parallel coordinates)")
    ax.axhline(0, color=INK, linewidth=0.5)
    ax.legend(frameon=False)
    return _save(fig, path)


def optimization_figures(
    history: Sequence[Mapping[str, Any]],
    front: Mapping[str, Any] | None,
    output: Path,
    *,
    font_size: float = FONT_SIZE,
) -> list[Path]:
    """Write every figure of an optimization in ``output``; return the PNG paths."""
    written = [objective_figure(history, output / "objective", font_size=font_size)]
    if front is not None and front.get("objectives"):
        written.append(front_figure(front, output / "front", font_size=font_size))
        written.append(hypervolume_figure(history, output / "hypervolume", font_size=font_size))
    return written
