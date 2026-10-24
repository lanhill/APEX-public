"""Cross validation for iterative ensemble smoothers.

Importing this package registers the stopping rules ``patience`` and ``select`` in
:data:`apex.core.plugins.stopping_rules`. :mod:`.data_split` holds the data-split
explorer's computations (Luo & Ranazzi, Steps 1-4).
"""

from apex.algorithms.validation.stopping import PatienceStopping, SelectStopping

__all__ = ["PatienceStopping", "SelectStopping"]
