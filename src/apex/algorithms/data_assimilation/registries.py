"""Plugin registries for ensemble data assimilation."""

from typing import TYPE_CHECKING

from apex.core.registry import Registry

if TYPE_CHECKING:
    from apex.algorithms.data_assimilation.algorithm import EnsembleAlgorithm
    from apex.algorithms.data_assimilation.chop.components import ChopComponent
    from apex.algorithms.data_assimilation.gmac.constraints import (
        ConstraintFunction,
        ConstraintMetric,
    )
    from apex.algorithms.data_assimilation.gmac.parts import GMACNorm, GMACTransform
    from apex.algorithms.data_assimilation.localization import Localization
    from apex.algorithms.data_assimilation.protocols import DampingScheme, UpdateStrategy

#: How an analysis inverts the ensemble-approximated sensitivity (``"approx"``, ...).
update_strategies: "Registry[UpdateStrategy]" = Registry("update_strategies")

#: Ensemble algorithms the generic driver runs (``"esmda"``, ``"gies"``, ...).
methods: "Registry[EnsembleAlgorithm]" = Registry("methods")

#: Deprecated alias of :data:`methods` (the two method registries were merged);
#: kept for one release. It is the same registry: whatever registers here must implement
#: ``EnsembleAlgorithm`` (a plugin for the removed protocols fails when the run starts).
iterative_methods = methods

#: How iterative methods adapt their damping (``"rlmmac"``, ``"lm"``).
damping_schemes: "Registry[DampingScheme]" = Registry("damping_schemes")

#: Localization methods for update strategies (``"adaptive"``, ``"distance"``, ...).
localizations: "Registry[Localization]" = Registry("localizations")

#: GMAC's model transforms S_k of its metric terms (``"identity"``, ``"tv"``, ...).
gmac_transforms: "Registry[GMACTransform]" = Registry("gmac_transforms")

#: GMAC's norms, whose Hessians make a term's metric (``"l2sq"``, ``"l1sq"``, ``"lpq"``).
gmac_norms: "Registry[GMACNorm]" = Registry("gmac_norms")

#: C-GIES's constraint functions, equality or inequality (``"box"``, ``"histogram"``, ...).
#:
gmac_constraints: "Registry[ConstraintFunction]" = Registry("gmac_constraints")

#: C-GIES's distance metrics of constraint residuals (``"log_barrier"``, ``"log_channel"``,
#: ``"quadratic"``, ``"l1sq"``).
gmac_constraint_metrics: "Registry[ConstraintMetric]" = Registry("gmac_constraint_metrics")

#: Hyper-parameterized components whose per-member parameters CHOP estimates
#: (``"length_scale"``, ``"gain_inflation"``, ``"data_weights"``).
chop_components: "Registry[ChopComponent]" = Registry("chop_components")
