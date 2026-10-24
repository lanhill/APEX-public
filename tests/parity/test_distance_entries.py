"""Upstream PET's distance localization, ``distance_loc`` (``e65fb10``).

The reference (``distance_entries_masks``) builds PET's taper from per-(data, time,
parameter) entries on a 10 x 8 x 2 grid with two parameters, over all cells and over the
active cells of an ``actnum``, and the approx steps it localizes at lambda 0 and 1.5.
APEX's ``distance_entries`` gets the same entries in its own form (1-based Eclipse
positions; the generator converts them to PET's 0-based ``x = j - 1``, ``y = i - 1``).
"""

import json
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from apex.algorithms.data_assimilation import (
    ApproxUpdate,
    DataErrorCovariance,
    EntryDistanceLocalization,
)
from apex.algorithms.data_assimilation.entry_localization import planar_taper
from apex.domain import DataKey

type ReferenceLoader = Callable[[str], dict[str, NDArray[np.generic]]]

#: PET places a cropped stencil; APEX evaluates the same formulas at the same offsets
#: (measured 2e-15, from rotation rounding).
TAPER_ATOL = 1e-13
#: As the other approx steps: SVD and summation order (measured 2e-15).
STEP_RTOL = 1e-12


def _floats(ref: dict[str, NDArray[np.generic]], key: str) -> NDArray[np.float64]:
    return np.asarray(ref[key], dtype=np.float64)


def _keys(ref: dict[str, NDArray[np.generic]]) -> list[DataKey]:
    components = int(np.asarray(ref["in_seis_components"]))
    keys: list[DataKey] = []
    for time in np.asarray(ref["in_times"]).tolist():
        for datatype in np.asarray(ref["in_datatypes"]).tolist():
            if datatype == "SEIS":
                keys.extend(DataKey(time, datatype, n) for n in range(components))
            else:
                keys.append(DataKey(time, datatype))
    return keys


def _localization(
    ref: dict[str, NDArray[np.generic]], layout: str, tmp_path: Path
) -> EntryDistanceLocalization:
    grid = [int(n) for n in np.asarray(ref["in_grid"])]
    actnum_file = None
    cells = int(np.prod(grid))
    if layout == "actnum":
        actnum_file = tmp_path / "actnum.npy"
        np.save(actnum_file, np.asarray(ref["in_actnum"]))
        cells = int(np.asarray(ref["in_actnum"]).sum())
    return EntryDistanceLocalization(
        grid,
        json.loads(str(ref["in_entries"])),
        variable_names=["PERMX", "PORO"],
        block_sizes=[cells, cells],
        actnum_file=actnum_file,
    )


@pytest.mark.pet_parity
@pytest.mark.parametrize("layout", ["full", "actnum"])
def test_the_taper_matches_pets(
    pet_reference: ReferenceLoader, layout: str, tmp_path: Path
) -> None:
    ref = pet_reference("distance_entries_masks")
    localization = _localization(ref, layout, tmp_path)
    state = _floats(ref, f"in_state_{layout}")

    rng = np.random.default_rng(0)  # unused: the taper is deterministic
    localization.prepare(state, _floats(ref, f"in_predicted_{layout}"), _keys(ref), rng)

    assert localization.taper_matrix is not None
    np.testing.assert_allclose(
        localization.taper_matrix, _floats(ref, f"out_taper_{layout}"), rtol=0, atol=TAPER_ATOL
    )


@pytest.mark.pet_parity
@pytest.mark.parametrize("layout", ["full", "actnum"])
@pytest.mark.parametrize("damping", [0.0, 1.5])
def test_the_localized_step_matches_pets(
    pet_reference: ReferenceLoader, layout: str, damping: float, tmp_path: Path
) -> None:
    ref = pet_reference("distance_entries_masks")
    error = DataErrorCovariance.diagonal(_floats(ref, f"in_variances_{layout}"))
    perturbed = _floats(ref, f"in_observations_{layout}")[:, None] + error.correlate(
        _floats(ref, f"in_z_{layout}")
    )
    np.testing.assert_array_equal(perturbed, ref[f"out_perturbed_observations_{layout}"])
    state, predicted = _floats(ref, f"in_state_{layout}"), _floats(ref, f"in_predicted_{layout}")
    localization = _localization(ref, layout, tmp_path)
    rng = np.random.default_rng(0)  # unused: the taper is deterministic
    localization.prepare(state, predicted, _keys(ref), rng)
    strategy = ApproxUpdate(float(_floats(ref, "in_energy")), localization=localization)

    step = strategy.step(state, predicted, perturbed, error, damping=damping, rng=rng)

    expected = _floats(ref, f"out_step_{layout}_lambda_{damping:g}")
    np.testing.assert_allclose(step, expected, rtol=0, atol=STEP_RTOL * np.abs(expected).max())


@pytest.mark.pet_parity
def test_pets_asymmetric_crop_shifts_its_kernel_by_one_cell(
    pet_reference: ReferenceLoader,
) -> None:
    """A recorded PET defect (pet_parity.md), not reproduced: APEX centres the kernel.

    PET's taper equals APEX's evaluated one cell lower in ``j``; APEX's own taper is
    centred (``tests/unit/algorithms/test_da_entry_localization.py``).
    """
    ref = pet_reference("distance_entries_miscentred")
    nx, ny, _ = (int(n) for n in np.asarray(ref["in_grid"]))
    i, j, _ = (int(n) for n in np.asarray(ref["in_position"]))
    ratio, angle = (float(v) for v in _floats(ref, "in_anisotropy"))
    radius = float(_floats(ref, "in_radius"))

    def apex(centre_j: int) -> NDArray[np.float64]:
        return planar_taper(
            (ny, nx), (i - 1, centre_j - 1), "gaspari_cohn", radius, (ratio, angle), 20
        ).ravel()

    pet = _floats(ref, "out_taper")
    assert np.abs(apex(j) - pet).max() > 0.1  # PET is off centre
    np.testing.assert_allclose(apex(j - 1), pet, rtol=0, atol=TAPER_ATOL)  # by one cell in j
