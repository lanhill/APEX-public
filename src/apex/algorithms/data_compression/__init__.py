"""Data compression: transforms of simulated data before the analysis.

Importing this package registers the ``wavelet`` transform in
:data:`apex.core.plugins.data_transforms`.
"""

from apex.algorithms.data_compression.transform import WaveletCompression
from apex.algorithms.data_compression.wavelet import (
    FieldCompression,
    WaveletOptions,
    decompose,
    fit_field,
    recompose,
)

__all__ = [
    "FieldCompression",
    "WaveletCompression",
    "WaveletOptions",
    "decompose",
    "fit_field",
    "recompose",
]
