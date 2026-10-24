"""The ensemble smoother (ES): one update with all the data.

Ported from PET
---------------
PET module : src/pipt/update_schemes/es.py (ES), src/pipt/update_schemes/enkf.py
             (EnKF.calc_analysis), src/pipt/update_schemes/analysis/approx.py
PET commit : e65fb1058411c8dad7ca70a09d501e0b18075e75
Reference  : Evensen, G. (2009). Data Assimilation: The Ensemble Kalman Filter (2nd ed.).
             Springer. van Leeuwen, P. J. & Evensen, G. (1996). Data assimilation and
             inverse methods in terms of a probabilistic formulation. Monthly Weather
             Review, 124, 2898-2913.
Differences: none in the update. PET's ES perturbs the observations with the un-inflated
             ``C_d`` and runs the approx analysis at ``lambda = 0``: ES-MDA's step with a
             single ``alpha = 1``, which this class is. PET scores the prior against its
             first draw and the posterior against the redrawn perturbation it assimilated,
             so its reduction mixes two draws (and is optimistic); APEX, as for ES-MDA,
             scores both against one separate un-inflated draw (PET's ``enObs_conv``).
             PET's ES maps the ``full`` flavour onto ``approx`` (they coincide in one
             step); so does ``apex pet-import``.

``m_j <- m_j + C_md (C_dd + C_d)^-1 (d_j - g(m_j))``, one step, with ``d_j`` drawn from
``N(d_obs, C_d)``.
"""

from numpy.typing import ArrayLike

from apex.algorithms.data_assimilation.algorithm import Status
from apex.algorithms.data_assimilation.esmda import ESMDA
from apex.algorithms.data_assimilation.protocols import UpdateStrategy
from apex.algorithms.data_assimilation.registries import methods


@methods.register("es")
class ES(ESMDA):
    """Ensemble smoother: one update with all the data, ES-MDA's single step at ``alpha = 1``.

    Parameters
    ----------
    strategy
        Update strategy; ``ApproxUpdate()`` by default (PET ``analysis = "approx"``).
    lower_bounds, upper_bounds
        Optional per-state bounds ``(n_state,)`` applied to the posterior.
    """

    status: Status = "parity"

    def __init__(
        self,
        strategy: UpdateStrategy | None = None,
        lower_bounds: ArrayLike | None = None,
        upper_bounds: ArrayLike | None = None,
    ) -> None:
        super().__init__((1.0,), strategy, lower_bounds, upper_bounds)
