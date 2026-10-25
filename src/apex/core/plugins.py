"""Plugin registries: simulators, executors, priors, transforms, metrics, rules, scores, ....

Also LLM providers, run summarizers, figures and figure presets, and the converter and
geomodeling skills.

"""

from apex.core.conversion import ConverterSkill
from apex.core.engines import RunEngine
from apex.core.figures import Figure, FigurePreset
from apex.core.forecast import ForecastScore
from apex.core.geomodeling import GeomodelingSkill, VariogramEstimator
from apex.core.llm import LLMProvider
from apex.core.maintenance import MaintenanceTask
from apex.core.metrics import PerformanceMetric
from apex.core.registry import Registry
from apex.core.simulation import (
    DataTransform,
    Executor,
    JointPriorGenerator,
    ModelErrorModel,
    Parameterization,
    PriorGenerator,
    Simulator,
)
from apex.core.summaries import Summarizer
from apex.core.validation import StoppingRule

#: Forward simulators by name (``"linear"``, ``"opm_flow"``, ...).
simulators: Registry[Simulator] = Registry("simulators")

#: Ensemble executors by name (``"serial"``, ``"process"``, ...).
executors: Registry[Executor] = Registry("executors")

#: The service's maintenance tasks (entry-point group ``apex.maintenance``).
maintenance_tasks: Registry[MaintenanceTask] = Registry("maintenance")

#: Initial-ensemble generators by name (``"gaussian"``, ``"file"``, ``"gaussian_field"``, ...).
priors: Registry[PriorGenerator | JointPriorGenerator] = Registry("priors")

#: Maps from the state to the simulator's properties (``"pgs_truncation"``).
parameterizations: Registry[Parameterization] = Registry("parameterizations")

#: Model-error (MEC) models added to a simulator's output (``"pg_lstm"``).
model_error_models: Registry[ModelErrorModel] = Registry("model_error_models")

#: Transforms of simulated data before the analysis (``"wavelet"``, ...).
data_transforms: Registry[DataTransform] = Registry("data_transforms")

#: Performance metrics evaluated at every forecast (``"normalized_data_mismatch"``, ``"rmse"``,
#: ...).
metrics: Registry[PerformanceMetric] = Registry("metrics")

#: Cross-validation stopping rules (``"patience"``, ``"select"``).
stopping_rules: Registry[StoppingRule] = Registry("stopping_rules")

#: Scores of a forecast against the truth's (``"crps"``, ``"coverage_p10_p90"``, ...).
forecast_scores: Registry[ForecastScore] = Registry("forecast_scores")

#: Language-model providers for runtime agents (``"anthropic"``, ``"openai_compatible"``,
#: ``"ollama"``, ``"stub"``).
llm_providers: Registry[LLMProvider] = Registry("llm_providers")

#: Run summarizers by workflow (``"history_matching"``; reports across runs).
summaries: Registry[Summarizer] = Registry("summaries")

#: Run engines besides APEX's own (``"ert"`` with the ``ert`` extra), registered by
#: their packages through the entry-point group ``apex.engines``.
engines: Registry[RunEngine] = Registry("engines")

#: Figure kinds by name (``"metric_distribution"``, ``"data_bands"``, ...).
figures: Registry[Figure] = Registry("figures")

#: Named sets of figure requests (``"brugge_classic"``).
figure_presets: Registry[FigurePreset] = Registry("figure_presets")

#: Converter skills by name (``"opm_flow"``, ``"eclipse"``, ``"pet"``, ...), also
#: registered by other packages through the entry-point group ``apex.converter_skills``.
converter_skills: Registry[ConverterSkill] = Registry("converter_skills")

#: Geomodeling skills by name (``"continuous"``, ``"facies_pluri"``), also
#: registered by other packages through the entry-point group ``apex.geomodeling_skills``.
geomodeling_skills: Registry[GeomodelingSkill] = Registry("geomodeling_skills")

#: Variogram estimators by name (``"wls"``): a variogram from hard data.
variogram_estimators: Registry[VariogramEstimator] = Registry("variogram_estimators")
