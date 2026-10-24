"""The data-split explorer: how to split production data for cross validation.

Use it before a run to choose the ``[validation] cv`` wells. It suggests and never
decides; the configuration stays explicit, so a run is reproducible:

>>> from apex.workflows.data_split import explore
>>> report = explore(observations)                  # doctest: +SKIP
>>> print(report.candidates[1].snippet())           # doctest: +SKIP

The command-line form, ``apex data-split CONFIG``, also writes the tables and figures.
Field data (e.g. seismic) always stay in history matching.
"""

import logging
from collections.abc import Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

from apex.algorithms.validation.data_split import explore as _explore
from apex.domain.data_split import SplitExploration
from apex.domain.observations import DataKey, ObservationSet
from apex.storage.run_directory import RunDirectory

logger = logging.getLogger(__name__)


def explore(
    observations: ObservationSet,
    *,
    prior_keys: Sequence[DataKey] | None = None,
    prior_predicted: NDArray[np.float64] | None = None,
    max_cv: int | None = None,
) -> SplitExploration:
    """Rank the wells as CV candidates and propose splits.

    Parameters
    ----------
    observations
        The raw observations of the case.
    prior_keys, prior_predicted
        Optionally, a prior forecast ``(n_data, n_members)`` in the analysis data space
        (see :func:`prior_forecast`), for the prior-ensemble redundancy of each well.
    max_cv
        Largest candidate CV set, in wells; default all wells but one.

    Returns
    -------
    SplitExploration
        Groups, correlations, per-well scores, redundancy and candidate splits.
    """
    report = _explore(
        observations, prior_keys=prior_keys, prior_predicted=prior_predicted, max_cv=max_cv
    )
    logger.info(
        "data split explored",
        extra={"event": "data_split_explored", "scores": report.scores, "notes": report.notes},
    )
    return report


def prior_forecast(
    run_dir: RunDirectory, keys: Mapping[str, Sequence[DataKey]]
) -> tuple[list[DataKey], NDArray[np.float64]]:
    """Return a run's prior forecast (``iter_0``) and its keys, held-out rows included.

    Parameters
    ----------
    run_dir
        A run that has forecast its prior.
    keys
        The run's keys per role: ``"history_matching"`` (the rows of ``predicted``) and,
        if it held data out, ``"cv"`` and ``"test"`` (``cv_predicted``, ``test_predicted``).

    Raises
    ------
    FileNotFoundError
        If the run has not saved its prior forecast.
    """
    saved = run_dir.load_arrays("iter_0")
    all_keys = list(keys["history_matching"])
    blocks = [saved["predicted"]]
    for role in ("cv", "test"):
        if keys.get(role):
            all_keys.extend(keys[role])
            blocks.append(saved[f"{role}_predicted"])
    return all_keys, np.vstack(blocks)
