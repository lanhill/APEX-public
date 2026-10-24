"""Grid coarsening against SimulatorWrap (``subsurface`` @ ``d29dc57``).

The reference (``upscaling_records``) runs SimulatorWrap's ``_unif``/``_Haar`` and
``write_coarse``; ``_Haar`` through a shim that makes its two float slices integer (its
code cannot run on Python 3 otherwise). The records must be identical.
"""

from collections.abc import Callable

import numpy as np
import pytest
from numpy.typing import NDArray

from apex.simulation.upscaling import coarsen

type ReferenceLoader = Callable[[str], dict[str, NDArray[np.generic]]]

#: As the generator's _UPSCALING: (method, dim, max_trunc, max_diff, wells, radii).
CASES = {
    "unif_half": ("uniform", (16, 12), 0.5, 0.0, [[3, 4], [10, 13]], [1, 2]),
    "unif_all": ("uniform", (13, 10), 1.0, 0.0, [[5, 5]], [1]),
    "haar_dyadic": ("haar", (16, 16), 0.6, 0.15, [[4, 4]], [1]),
    "haar_odd": ("haar", (12, 10), 0.6, 0.15, [], []),
}


@pytest.mark.pet_parity
@pytest.mark.parametrize("name", list(CASES))
def test_coarsen_records_match_simulatorwrap(pet_reference: ReferenceLoader, name: str) -> None:
    ref = pet_reference("upscaling_records")
    method, dim, max_trunc, max_diff, wells, radius = CASES[name]

    records = coarsen(
        np.asarray(ref[f"in_{name}_field"], dtype=np.float64),
        dim,
        method=method,  # type: ignore[arg-type]
        max_trunc=max_trunc,
        max_diff=max_diff,
        wells=wells,
        radius=radius,
    )

    expected = np.asarray(ref[f"out_{name}_records"], dtype=np.int64)
    assert len(expected) > 0
    np.testing.assert_array_equal(np.array(records, dtype=np.int64).reshape(-1, 9), expected)


@pytest.mark.pet_parity
def test_no_truncation_means_no_coarsening(pet_reference: ReferenceLoader) -> None:
    """SimulatorWrap's run_fwd_sim skips coarsening when maxtrunc is 0; write_coarse
    alone (the reference) would still merge 2 x 2 blocks."""
    ref = pet_reference("upscaling_records")
    assert len(ref["out_unif_none_records"]) > 0
    assert coarsen(np.asarray(ref["in_unif_none_field"], float), (8, 8), max_trunc=0.0) == []


def test_layers_get_the_same_blocks_and_keep_their_cells() -> None:
    """An APEX extension: SimulatorWrap coarsens one layer; with nz layers every layer is
    merged in the same x-y blocks (K1 = 1, K2 = NZ = nz), following the layer mean."""
    rng = np.random.default_rng(3)
    layer = rng.uniform(size=(8, 8))
    field = np.concatenate([layer.reshape(-1), (layer + 0.1).reshape(-1)])

    records = coarsen(field, (8, 8), method="haar", max_trunc=0.9, max_diff=0.5, layers=2)

    single = coarsen(layer.reshape(-1) + 0.05, (8, 8), method="haar", max_trunc=0.9, max_diff=0.5)
    assert records == [(*r[:4], 1, 2, 1, 1, 2) for r in single]
    with pytest.raises(ValueError, match="8 x 8 x 3"):
        coarsen(field, (8, 8), max_trunc=0.5, layers=3)
