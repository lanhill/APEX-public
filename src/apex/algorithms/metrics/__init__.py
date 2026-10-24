"""Performance metrics and forecast scores.

Importing this package registers ``normalized_data_mismatch``, ``rmse``, ``ssim`` and
``constraint_cost`` in
:data:`apex.core.plugins.metrics`, and ``coverage_p10_p90``, ``crps``, ``mean_error`` and
``spread`` in :data:`apex.core.plugins.forecast_scores`.
"""

from apex.algorithms.metrics.constraints import ConstraintCost
from apex.algorithms.metrics.forecast import CRPS, CoverageP10P90, MeanError, Spread
from apex.algorithms.metrics.kernels import normalized_l2, ssim
from apex.algorithms.metrics.performance import ModelRMSE, ModelSSIM, NormalizedDataMismatch

__all__ = [
    "CRPS",
    "ConstraintCost",
    "CoverageP10P90",
    "MeanError",
    "ModelRMSE",
    "ModelSSIM",
    "NormalizedDataMismatch",
    "Spread",
    "normalized_l2",
    "ssim",
]
