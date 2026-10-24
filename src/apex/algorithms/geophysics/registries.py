"""Plugin registries of the geophysical forward models."""

from typing import TYPE_CHECKING

from apex.core.registry import Registry

if TYPE_CHECKING:
    from apex.algorithms.geophysics.attributes import AVOAttribute
    from apex.algorithms.geophysics.displacement import DisplacementKernel
    from apex.algorithms.geophysics.petro_elastic import PetroElasticModel
    from apex.algorithms.geophysics.reflectivity import PPReflectivity

#: PP reflection coefficients of an elastic interface (``"zoeppritz"``, ``"aki_richards"``,
#: ``"fatti"``, ``"shuey"``, ``"approx_zoeppritz_pp"``).
reflectivity: "Registry[PPReflectivity]" = Registry("reflectivity")

#: Petro-elastic models: Vp, Vs and density from a cell's flow state (``"standard"``,
#: ``"soft_sand"``).
petro_elastic_models: "Registry[PetroElasticModel]" = Registry("petro_elastic_models")

#: AVO attributes: a map value per trace over a time window, or the trace (``"rms"``,
#: ``"mean"``, ``"min"``, ``"max"``, ``"sum_negative"``, ``"trace"``),
#: or the time shift to the reservoir's base (``"time_shift"``; revision 1).
avo_attributes: "Registry[AVOAttribute]" = Registry("avo_attributes")

#: Vertical displacement at stations from cells' volume changes (``"geertsma"``).
displacement_kernels: "Registry[DisplacementKernel]" = Registry("displacement_kernels")
