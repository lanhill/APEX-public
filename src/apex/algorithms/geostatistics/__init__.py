"""Geostatistical kernels: variogram models, grid covariances and correlated sampling.

Ported from PET
---------------
PET module : geostat/decomp.py (Cholesky.gen_cov2d, variogram_model, _edist2d, gen_real) of
             github.com/Python-Ensemble-Toolbox/Geostatistics, used by upstream PET's
             src/misc/structures/layout.py (StateLayout.from_prior_info)
PET commit : Geostatistics 3f9f0c876815db140fae3404d322892190cb6728 (pinned by PET
             e65fb1058411c8dad7ca70a09d501e0b18075e75)
Reference  : Chiles, J.-P. & Delfiner, P. (2012). Geostatistics: Modeling Spatial
             Uncertainty, 2nd ed., Wiley (variogram models, p. 641).
Differences:
- **Rotation.** PET builds its anisotropy transform as ``rescale * rotate`` element-wise,
  which drops the sine terms, so ``angle`` only divides the ranges by ``|cos(angle)|`` and
  never rotates. APEX uses the matrix product.
- **Cell order.** PET enumerates cells with ``np.mgrid[1:nx+1, 1:ny+1]``, so y varies
  fastest, while states are in Eclipse order (x fastest). On square grids this transposes
  the field; on non-square grids it scrambles it. APEX uses Eclipse order. On square
  grids PET's field is reproduced exactly by the ranges and azimuth from
  ``apex.storage.pet_config.pet_field_geometry`` (the parity cases).
- **Names.** PET's ``'exp'`` is ``exp(-3 u^1.9)``, not exponential; it is APEX's
  ``"stable"`` with ``exponent = 1.9``. APEX's ``"exponential"`` is ``exp(-3 u)``.
- **Ranges.** Anisotropy is ``major_range`` / ``minor_range`` plus an ``azimuth`` (degrees,
  counter-clockwise from the i-axis). PET's ``range`` and ``aniso`` correspond to
  ``minor_range = range`` and ``major_range = aniso * range``.
- **Conditioning** to hard data (not in PET) is by kriging the residuals:
  :mod:`apex.algorithms.geostatistics.conditioning`.
- **Singular covariances** (e.g. Gaussian variograms) fall back to a clipped
  eigen-decomposition with a logged warning; PET's Cholesky fails.
"""

from apex.algorithms.geostatistics.conditioning import (
    condition_by_kriging,
    kriged_excursion,
    kriging_factor,
    layer_cross_covariance,
)
from apex.algorithms.geostatistics.covariance import (
    VARIOGRAM_MODELS,
    correlated_sample,
    covariance_factor,
    layer_covariance,
    variogram_covariance,
)

__all__ = [
    "VARIOGRAM_MODELS",
    "condition_by_kriging",
    "correlated_sample",
    "covariance_factor",
    "kriged_excursion",
    "kriging_factor",
    "layer_covariance",
    "layer_cross_covariance",
    "variogram_covariance",
]
