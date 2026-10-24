"""Scores of an ensemble forecast against a reference forecast.

A :class:`ForecastScore` compares the ensemble's forecast of one vector over the
prediction period with the true model's forecast (synthetic cases), e.g. how often the
truth lies within the ensemble's 10th-90th percentile band. Scores are plugins in
:data:`apex.core.plugins.forecast_scores`.
"""

from typing import Protocol

import numpy as np
from numpy.typing import NDArray


class ForecastScore(Protocol):
    """One forecast score."""

    def score(self, ensemble: NDArray[np.float64], truth: NDArray[np.float64]) -> float:
        """Score ``ensemble`` ``(n_times, n_members)`` against ``truth`` ``(n_times,)``.

        Raises
        ------
        ValueError
            If the shapes do not match or there are no times or members.
        """
        ...
