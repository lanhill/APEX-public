"""Facies modelling: truncation rules, plurigaussian priors and parameterizations.

Importing this package registers ``gstools_pgs`` (prior), ``pgs_truncation`` and
``facies_distance`` (parameterizations). GSTools is imported only when a prior draws fields;
the SNESIM prior (``mpslib_snesim``, :mod:`apex.algorithms.facies.snesim`) runs MPSlib's program.
"""

from apex.algorithms.facies.distance import (
    FaciesDistance,
    FaciesProperties,
    facies_from_distances,
    signed_distances,
)
from apex.algorithms.facies.pgs import PGSPrior, PGSTruncation, load_rule
from apex.algorithms.facies.rules import LithotypeRule, condition_latent
from apex.algorithms.facies.snesim import MpslibSnesim

__all__ = [
    "FaciesDistance",
    "FaciesProperties",
    "LithotypeRule",
    "MpslibSnesim",
    "PGSPrior",
    "PGSTruncation",
    "condition_latent",
    "facies_from_distances",
    "load_rule",
    "signed_distances",
]
