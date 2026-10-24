"""CHOP: continuous hyper-parameter optimization of an iterative smoother.

- :mod:`.components`: the hyper-parameterized components (registry ``chop_components``);
- :mod:`.taper`: the per-member tapers of the model and hyper-parameter updates;
- :mod:`.method`: the method ``chop``, which wraps an iterative method on GIES's loop.
"""

from apex.algorithms.data_assimilation.chop import components, method, taper

__all__ = ["components", "method", "taper"]
