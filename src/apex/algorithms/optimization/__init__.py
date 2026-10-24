"""Ensemble optimization: PET's popt as optimizer, objective and step-rule plugins.

Importing this package registers its built-in plugins in :mod:`.registries`.
"""

from apex.algorithms.optimization.base import (
    OptimizationResult,
    OptimizerBase,
    StepReport,
    external_penalty,
)
from apex.algorithms.optimization.cma import CMA
from apex.algorithms.optimization.controls import ControlScaling, ControlSchedule
from apex.algorithms.optimization.deterministic import LineSearch, TrustRegion
from apex.algorithms.optimization.ecalc import (
    EcalcEmissions,
    EcalcEmissionTotal,
    EcalcNetPresentValue,
    EcalcParetoNetPresentValue,
    RateTable,
    RenewableCO2NetPresentValue,
)
from apex.algorithms.optimization.economics import NetPresentValue, RenewableNetPresentValue
from apex.algorithms.optimization.enopt import EnOpt
from apex.algorithms.optimization.ensemble_gradient import (
    GaussianEnsemble,
    ensemble_gradient,
    ensemble_hessian,
    gaussian_perturbations,
    smc_weights,
)
from apex.algorithms.optimization.ga import GeneticAlgorithm
from apex.algorithms.optimization.generalized import GeneralizedEnsemble, beta_concentration
from apex.algorithms.optimization.genopt import GenOpt
from apex.algorithms.optimization.moead import MOEAD
from apex.algorithms.optimization.nsga2 import NSGA2
from apex.algorithms.optimization.nsga3 import NSGA3
from apex.algorithms.optimization.objectives import Ackley, Quadratic, Rastrigin, Rosenbrock
from apex.algorithms.optimization.protocols import (
    ConstrainedObjective,
    DifferentiableObjective,
    Draws,
    GeneratorDraws,
    Objective,
    Problem,
    ResponseObjective,
    VectorObjective,
)
from apex.algorithms.optimization.pso import ParticleSwarm
from apex.algorithms.optimization.registries import (
    objectives,
    optimizers,
    response_objectives,
    step_rules,
)
from apex.algorithms.optimization.smcopt import SmcOpt
from apex.algorithms.optimization.step_rules import Adam, AdaMax, GradientDescent, Steihaug
from apex.algorithms.optimization.test_problems import (
    BinhKorn,
    Dtlz1,
    Dtlz2,
    Fonseca,
    Kursawe,
    Schaffer,
    Srinivas,
    Tanaka,
    Zdt1,
    Zdt2,
    Zdt3,
)

__all__ = [
    "CMA",
    "MOEAD",
    "NSGA2",
    "NSGA3",
    "Ackley",
    "AdaMax",
    "Adam",
    "BinhKorn",
    "ConstrainedObjective",
    "ControlScaling",
    "ControlSchedule",
    "DifferentiableObjective",
    "Draws",
    "Dtlz1",
    "Dtlz2",
    "EcalcEmissionTotal",
    "EcalcEmissions",
    "EcalcNetPresentValue",
    "EcalcParetoNetPresentValue",
    "EnOpt",
    "Fonseca",
    "GaussianEnsemble",
    "GenOpt",
    "GeneralizedEnsemble",
    "GeneratorDraws",
    "GeneticAlgorithm",
    "GradientDescent",
    "Kursawe",
    "LineSearch",
    "NetPresentValue",
    "Objective",
    "OptimizationResult",
    "OptimizerBase",
    "ParticleSwarm",
    "Problem",
    "Quadratic",
    "Rastrigin",
    "RateTable",
    "RenewableCO2NetPresentValue",
    "RenewableNetPresentValue",
    "ResponseObjective",
    "Rosenbrock",
    "Schaffer",
    "SmcOpt",
    "Srinivas",
    "Steihaug",
    "StepReport",
    "Tanaka",
    "TrustRegion",
    "VectorObjective",
    "Zdt1",
    "Zdt2",
    "Zdt3",
    "beta_concentration",
    "ensemble_gradient",
    "ensemble_hessian",
    "external_penalty",
    "gaussian_perturbations",
    "objectives",
    "optimizers",
    "response_objectives",
    "smc_weights",
    "step_rules",
]
