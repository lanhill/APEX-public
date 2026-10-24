"""The gaussian_field prior against PET's prior sampler (upstream e65fb10 + geostat 3f9f0c8)."""

from collections.abc import Callable

import numpy as np
import pytest
from numpy.typing import NDArray

from apex.algorithms.priors import GaussianFieldPrior
from apex.domain import StateVariable
from apex.storage.pet_config import pet_field_geometry

# tests/ is not an importable package, so spell out the fixture's type here.
type ReferenceLoader = Callable[[str], dict[str, NDArray[np.generic]]]


class _QueuedNormals:
    """Generator stand-in returning PET's per-layer normals in order."""

    def __init__(self, draws: NDArray[np.generic]) -> None:
        self._draws = [np.asarray(d, dtype=np.float64) for d in draws]

    def standard_normal(self, shape: tuple[int, int]) -> NDArray[np.float64]:
        draw = self._draws.pop(0)
        assert draw.shape == shape
        return draw


@pytest.mark.pet_parity
@pytest.mark.parametrize(
    ("case", "variogram", "exponent"),
    [
        ("prior_field_sph", "spherical", 1.0),
        ("prior_field_exp", "stable", 1.9),  # PET's "exp" is exp(-3 u^1.9)
        ("prior_field_cub", "cubic", 1.0),
        ("prior_field_aniso", "spherical", 1.0),  # aniso and angle, as `apex pet-import` maps
    ],
)
def test_gaussian_field_matches_pet_prior_sampler(
    pet_reference: ReferenceLoader, case: str, variogram: str, exponent: float
) -> None:
    ref = pet_reference(case)
    grid = [int(n) for n in ref["in_grid"]]
    geometry, _ = pet_field_geometry(
        grid[0],
        grid[1],
        [float(r) for r in ref["in_range"]],
        [float(a) for a in ref["in_aniso"]],
        [float(a) for a in ref["in_angle"]],
    )
    prior = GaussianFieldPrior(
        grid,
        mean=np.asarray(ref["in_mean"], dtype=np.float64),
        variance=[float(v) for v in ref["in_variance"]],
        variogram=variogram,
        exponent=exponent,
        **geometry,
    )

    field = prior.generate(
        StateVariable("permx", int(np.prod(grid))),
        ref["out_field"].shape[-1],
        _QueuedNormals(ref["in_z"]),  # type: ignore[arg-type]
    )

    # Measured <= 1e-14 absolute on values of O(5): rounding in the covariance (PET
    # scales distances, APEX ranges) carried through the Cholesky factor.
    expected = np.asarray(ref["out_field"], dtype=np.float64)
    np.testing.assert_allclose(field, expected, rtol=0.0, atol=5e-14)
