"""GMAC: GIES's generalised update with composable metric terms.

- :mod:`.parts`: the metric terms' transforms and norms (registries ``gmac_transforms``,
  ``gmac_norms``);
- :mod:`.constraints`: C-GIES's constraint functions and metrics (registries
  ``gmac_constraints``, ``gmac_constraint_metrics``);
- :mod:`.strategy`: the update strategy ``gmac``.
"""

from apex.algorithms.data_assimilation.gmac import constraints, parts, strategy

__all__ = ["constraints", "parts", "strategy"]
