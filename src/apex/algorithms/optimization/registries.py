"""Plugin registries for ensemble optimization."""

from typing import TYPE_CHECKING

from apex.core.registry import Registry

if TYPE_CHECKING:
    from apex.algorithms.optimization.base import OptimizerBase
    from apex.algorithms.optimization.protocols import Objective, ResponseObjective
    from apex.algorithms.optimization.step_rules import StepRule

#: Optimizers by name (``"enopt"``, ``"genopt"``, ``"smcopt"``, ``"line_search"``,
#: ``"trust_region"``).
optimizers: "Registry[OptimizerBase]" = Registry("optimizers")

#: Objectives by name (``"rosenbrock"``, ``"quadratic"``).
objectives: "Registry[Objective]" = Registry("objectives")

#: Objectives of a simulator's responses (``"npv"``), evaluated through the simulator the
#: optimization configures (phase 2).
response_objectives: "Registry[ResponseObjective]" = Registry("response_objectives")

#: Step rules of the ensemble optimizers (``"gd"``, ``"adam"``, ``"adamax"``, ``"steihaug"``).
step_rules: "Registry[StepRule]" = Registry("step_rules")
