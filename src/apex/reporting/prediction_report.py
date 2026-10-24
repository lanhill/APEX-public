"""Text tables of predictions, for ``apex report PRED_DIR``."""

from collections.abc import Mapping

import numpy as np

from apex.domain.prediction import PERCENTILE_LABELS, PredictionResult


def final_time_table(result: PredictionResult, convention: str = "percentile") -> str:
    """Each ensemble's forecast of every vector at the last time: members and percentiles."""
    columns = PERCENTILE_LABELS[convention]
    some = next(iter(result.forecasts.values()))
    header = f"{'vector':<16} {'ensemble':<14} {'n':>4} " + " ".join(
        f"{label:>12}" for label, _ in columns
    )
    if result.truth is not None:
        header += f" {'truth':>12}"
    lines = [f"forecast on {some.times[-1]}", header, "-" * len(header)]
    for v, vector in enumerate(some.vectors):
        for name, forecast in result.forecasts.items():
            values = forecast.values[v, -1, :]
            quantiles = np.percentile(values, [q for _, q in columns])
            line = f"{vector:<16} {name:<14} {values.size:>4} " + " ".join(
                f"{q:>12.5g}" for q in quantiles
            )
            if result.truth is not None:
                line += f" {result.truth[v, -1]:>12.5g}"
            lines.append(line)
    return "\n".join(lines)


def scores_table(scores: Mapping[str, Mapping[str, Mapping[str, float]]]) -> str | None:
    """Forecast scores per vector and ensemble, or ``None`` without scores."""
    if not scores:
        return None
    names = list(scores)
    score_names = list(next(iter(next(iter(scores.values())).values())))
    header = f"{'vector':<16} {'ensemble':<14} " + " ".join(f"{s:>16}" for s in score_names)
    lines = ["scores over the prediction period", header, "-" * len(header)]
    for vector in next(iter(scores.values())):
        for name in names:
            values = scores[name][vector]
            lines.append(
                f"{vector:<16} {name:<14} " + " ".join(f"{values[s]:>16.5g}" for s in score_names)
            )
    return "\n".join(lines)
