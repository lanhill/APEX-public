"""Ensemble data assimilation: error models, kernels, update strategies and methods.

Importing this package registers its built-in plugins in :mod:`.registries`.
"""

from apex.algorithms.data_assimilation.algorithm import (
    AlgorithmState,
    EnsembleAlgorithm,
    Estimate,
    Forecast,
    ForecastRequest,
    Forecasts,
    Record,
    RunSetup,
    Status,
    Step,
    keep,
    mean_model_request,
)
from apex.algorithms.data_assimilation.bounds import StateBounds
from apex.algorithms.data_assimilation.chop.method import Chop
from apex.algorithms.data_assimilation.data_error import DataErrorCovariance
from apex.algorithms.data_assimilation.entry_localization import EntryDistanceLocalization
from apex.algorithms.data_assimilation.es import ES
from apex.algorithms.data_assimilation.esmda import ESMDA
from apex.algorithms.data_assimilation.full_update import FullUpdate
from apex.algorithms.data_assimilation.gies import GIES, DampingSchedule, RLMMACUpdate
from apex.algorithms.data_assimilation.gmac.strategy import GMACUpdate
from apex.algorithms.data_assimilation.gn_enrml import GNEnRML, GNStepSchedule
from apex.algorithms.data_assimilation.lm_enrml import LMDampingSchedule, LMEnRML
from apex.algorithms.data_assimilation.local_analysis import (
    AdaptiveRowsUpdate,
    LocalAnalysisUpdate,
    LocalRLMMACUpdate,
)
from apex.algorithms.data_assimilation.localization import (
    AdaptiveLocalization,
    DistanceLocalization,
    Localization,
    PriorAdaptiveLocalization,
    ProjectedAdaptiveLocalization,
    ProjectedLocalization,
)
from apex.algorithms.data_assimilation.multilevel import HybridUpdate, MultilevelESMDA
from apex.algorithms.data_assimilation.protocols import (
    AnalysisResult,
    DampingDecision,
    DampingScheme,
    UpdateStrategy,
)
from apex.algorithms.data_assimilation.registries import (
    chop_components,
    damping_schemes,
    iterative_methods,
    localizations,
    methods,
    update_strategies,
)
from apex.algorithms.data_assimilation.subspace_update import SubspaceUpdate
from apex.algorithms.data_assimilation.transform_update import MarginalISUpdate, TransformUpdate
from apex.algorithms.data_assimilation.update_strategies import ApproxUpdate

__all__ = [
    "ES",
    "ESMDA",
    "GIES",
    "AdaptiveLocalization",
    "AdaptiveRowsUpdate",
    "AlgorithmState",
    "AnalysisResult",
    "ApproxUpdate",
    "Chop",
    "DampingDecision",
    "DampingSchedule",
    "DampingScheme",
    "DataErrorCovariance",
    "DistanceLocalization",
    "EnsembleAlgorithm",
    "EntryDistanceLocalization",
    "Estimate",
    "Forecast",
    "ForecastRequest",
    "Forecasts",
    "FullUpdate",
    "GMACUpdate",
    "GNEnRML",
    "GNStepSchedule",
    "HybridUpdate",
    "LMDampingSchedule",
    "LMEnRML",
    "LocalAnalysisUpdate",
    "LocalRLMMACUpdate",
    "Localization",
    "MarginalISUpdate",
    "MultilevelESMDA",
    "PriorAdaptiveLocalization",
    "ProjectedAdaptiveLocalization",
    "ProjectedLocalization",
    "RLMMACUpdate",
    "Record",
    "RunSetup",
    "StateBounds",
    "Status",
    "Step",
    "SubspaceUpdate",
    "TransformUpdate",
    "UpdateStrategy",
    "chop_components",
    "damping_schemes",
    "iterative_methods",
    "keep",
    "localizations",
    "mean_model_request",
    "methods",
    "update_strategies",
]
