"""Plain-text and JSON summaries of a history-matching run."""

from collections.abc import Sequence

from apex.domain.results import IterationRecord


def history_table(history: Sequence[IterationRecord], control: str = "damping") -> str:
    """Format the data mismatch history as an aligned text table.

    ``control`` heads the column of each trial's parameter (``IterationRecord.damping``):
    the method's own name for it, e.g. GN-EnRML's ``step length``.
    """
    header = (
        f"{'iter':>4}  {'data mismatch mean':>14}  {'data mismatch std*':>12}  "
        f"{'members':>7}  {'accepted':>8}  {control:>11}"
    )
    lines = [header, "-" * len(header)]
    for record in history:
        damping = "" if record.damping is None else f"{record.damping:.3g}"
        lines.append(
            f"{record.iteration:>4}  {record.data_mismatch_mean:>14.6g}  "
            f"{record.data_mismatch_std:>12.4g}  "
            f"{record.n_members:>7}  {'yes' if record.accepted else 'no':>8}  {damping:>11}"
        )
    # The methods' own statistic (PET's, which LM damping gates on); `apex summary` and the
    # API's step statistics give the sample std (ddof = 1) of every metric.
    lines.append(
        "* over the members: the population std (ddof = 0), as the methods use it; "
        "`apex summary` gives the sample std (ddof = 1)"
    )
    return "\n".join(lines)


def metrics_table(
    history: Sequence[IterationRecord], selected_iteration: int | None = None
) -> str | None:
    """Member means of each data type's data mismatch share and of every metric, prior vs final.

    ``final`` is the posterior: the last accepted step, or with cross validation the
    ``selected_iteration`` (the minimum-CV iterate). Returns ``None`` for runs
    that recorded neither (saved before they existed).
    """
    accepted = [r for r in history if r.accepted]
    if not accepted or (accepted[0].metrics is None and accepted[0].data_mismatch_by_type is None):
        return None
    prior, final = accepted[0], accepted[-1]
    if selected_iteration is not None:
        final = next(r for r in accepted if r.iteration == selected_iteration)
    rows: list[tuple[str, str, float, float]] = []
    # The data_mismatch metric has the per-type shares as its labels; older runs recorded
    # only the shares.
    shares = {} if "data_mismatch" in (prior.metrics or {}) else prior.data_mismatch_by_type
    for datatype, value in (shares or {}).items():
        rows.append(
            (
                "data_mismatch",
                datatype,
                value,
                (final.data_mismatch_by_type or {}).get(datatype, _NAN),
            )
        )
    # Keys of both: model_change has no prior value (it starts at the first update).
    first, last_metrics = prior.metrics or {}, final.metrics or {}
    for key in dict.fromkeys([*first, *last_metrics]):
        before, after = first.get(key, {}), last_metrics.get(key, {})
        rows.extend(
            (key, label, before.get(label, _NAN), after.get(label, _NAN))
            for label in dict.fromkeys([*before, *after])
        )
    width_metric = max(6, *(len(r[0]) for r in rows))
    width_label = max(5, *(len(r[1]) for r in rows))
    header = f"{'metric':<{width_metric}}  {'label':<{width_label}}  {'prior':>12}  {'final':>12}"
    lines = [
        f"metrics, member means (prior = iteration {prior.iteration}, final = {final.iteration}"
        + (", selected by cross validation)" if selected_iteration is not None else ")")
    ]
    lines += [header, "-" * len(header)]
    lines += [
        f"{m:<{width_metric}}  {label:<{width_label}}  {a:>12.6g}  {b:>12.6g}"
        for m, label, a, b in rows
    ]
    return "\n".join(lines)


_NAN = float("nan")


def run_summary(history: Sequence[IterationRecord], **fields: object) -> dict[str, object]:
    """Summarise a run: prior and final accepted data mismatch, reduction, and extra ``fields``.

    Raises
    ------
    ValueError
        If ``history`` is empty.
    """
    if not history:
        raise ValueError("a run summary needs at least the prior forecast")
    accepted = [record for record in history if record.accepted]
    prior, final = history[0], accepted[-1]
    reduction = (
        1.0 - final.data_mismatch_mean / prior.data_mismatch_mean
        if prior.data_mismatch_mean > 0
        else 0.0
    )
    return {
        **fields,
        "iterations": history[-1].iteration,
        "prior_data_mismatch_mean": prior.data_mismatch_mean,
        "final_data_mismatch_mean": final.data_mismatch_mean,
        "data_mismatch_reduction": reduction,
        "prior_metrics": prior.metrics,
        "final_metrics": final.metrics,
        "final_members": final.n_members,
    }
