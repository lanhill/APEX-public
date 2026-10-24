"""The metric ``constraint_cost``: a soft constraint's distance per member.

The values the user's paper reports for C-GIES (Luo & Cruz (2022), Computational
Geosciences 26(3), 571-594, doi:10.1007/s10596-022-10137-7; manuscript DASC_CG_R1.tex,
Tables 2 and 3: the
barrier and channel functions of the final ensembles), for any ensemble: each member's
``D(0 - c(m_j))`` for a constraint function ``c`` and metric ``D`` of the registries
``gmac_constraints`` and ``gmac_constraint_metrics``, and, for inequalities, the number of
violated rows. It sees the properties when a parameterization maps the state, as the
constraints do.
"""

from collections.abc import Mapping
from typing import Annotated, Any

import numpy as np
from numpy.typing import NDArray

from apex.algorithms.data_assimilation.gmac.constrained import DEFAULT_METRICS
from apex.algorithms.data_assimilation.gmac.constraints import (
    FUNCTION_ROLES,
    METRIC_ROLES,
    plugin_table,
)
from apex.algorithms.data_assimilation.gmac.parts import blocks
from apex.algorithms.data_assimilation.registries import gmac_constraint_metrics, gmac_constraints
from apex.core.errors import PluginError
from apex.core.metrics import MetricContext
from apex.core.plugins import metrics


@metrics.register("constraint_cost")
class ConstraintCost:
    """Each member's ``D(0 - c(m_j))`` and violated inequality rows.

    Parameters
    ----------
    function
        ``{name, options}`` of ``gmac_constraints`` (e.g. ``box``, ``histogram``).
    metric
        ``{name, options}`` of ``gmac_constraint_metrics``; default ``log_barrier`` for
        inequalities, ``log_channel`` for equalities.

    Labels: ``cost`` ``(n,)`` (``inf`` where a barrier is undefined), and ``violations``
    for inequalities.
    """

    requires_truth = False

    def __init__(
        self,
        function: Annotated[Mapping[str, Any], FUNCTION_ROLES],
        metric: Annotated[Mapping[str, Any], METRIC_ROLES] | None = None,
    ) -> None:
        try:
            f_name, f_options = plugin_table(function, "constraint_cost function")
            self.function = gmac_constraints.create(f_name, **f_options)
            m_name, m_options = plugin_table(
                metric or {"name": DEFAULT_METRICS[self.function.kind]}, "constraint_cost metric"
            )
            self.metric = gmac_constraint_metrics.create(m_name, **m_options)
        except (KeyError, OSError, PluginError, TypeError) as error:
            raise ValueError(f"constraint_cost: {error}") from error

    def evaluate(self, context: MetricContext) -> dict[str, NDArray[np.float64]]:
        """Return ``{"cost": (n,), "violations": (n,)}`` (no violations for equalities).

        Raises
        ------
        ValueError
            If the constraint names variables the ensemble lacks.
        """
        names = list(context.state)
        layout = blocks(
            [{"name": n, "size": context.state[n].shape[0], "grid": context.grids.get(n)}
             for n in names]
        )  # fmt: skip
        self.function.check(layout)
        stacked = np.concatenate([context.state[n] for n in names])
        values = self.function.values(stacked, layout)
        result = {"cost": np.asarray(self.metric.cost(-values), dtype=np.float64)}
        if self.function.kind == "inequality":
            result["violations"] = np.sum(values > 0, axis=0).astype(np.float64)
        return result
