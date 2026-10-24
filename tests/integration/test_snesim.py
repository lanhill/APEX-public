"""MPSlib's SNESIM through ``mpslib_snesim`` (opt-in, it runs a program).

A training image of channels (stripes four cells wide, half sand) on a 40 x 30 grid: the
hard data are honoured, the same seed gives the same realisations and different seeds
differ, and the sand proportion is the training image's within 0.1 (SNESIM reproduces
proportions only approximately: multigrids and a finite template).
"""

import os
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from apex.algorithms.facies import facies_from_distances
from apex.algorithms.facies.snesim import MpslibSnesim, bundled_program
from apex.domain.state import StateVariable

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(bundled_program() is None, reason="MPSlib (scikit-mps) is not installed"),
]


def test_snesim_honours_hard_data_and_reproduces_the_training_image(tmp_path: Path) -> None:
    i = np.arange(60)
    image = np.stack([(i // 4) % 2] * 60, axis=1)[:, :, None]
    np.save(tmp_path / "ti.npy", image)
    cells = [[3, 3, 1], [20, 15, 1], [38, 28, 1]]
    observed = [1, 0, 1]
    prior = MpslibSnesim(
        str(tmp_path / "ti.npy"),
        conditioning={"cells": cells, "facies": observed},
        template_size=[7, 7, 1],
    )
    variables = [StateVariable(name=f"d{k}", size=1200, grid=(40, 30, 1)) for k in range(2)]
    first = prior.generate_joint(variables, 4, np.random.default_rng(5))
    again = prior.generate_joint(variables, 4, np.random.default_rng(5))
    other = prior.generate_joint(variables, 4, np.random.default_rng(6))
    np.testing.assert_array_equal(first["d1"], again["d1"])
    assert not np.array_equal(first["d1"], other["d1"])
    for j in range(4):
        facies = facies_from_distances(np.stack([first["d0"][:, j], first["d1"][:, j]]))
        for (ci, cj, _), code in zip(cells, observed, strict=True):
            assert facies[(ci - 1) + 40 * (cj - 1)] == code
        assert abs(facies.mean() - 0.5) < 0.1
        # The stripes vary along i only: facies change more often along i than along j.
        grid = facies.reshape((40, 30), order="F")
        along_i = np.mean(grid[1:, :] != grid[:-1, :])
        along_j = np.mean(grid[:, 1:] != grid[:, :-1])
        assert along_i > 2 * along_j


def test_snesim_places_hard_data_with_an_anisotropic_cell_size(tmp_path: Path) -> None:
    i = np.arange(40)
    np.save(tmp_path / "ti.npy", np.stack([(i // 3) % 2] * 40, axis=1)[:, :, None])
    cells, observed = [[4, 9, 1], [17, 2, 1], [25, 18, 1]], [1, 1, 0]
    prior = MpslibSnesim(
        str(tmp_path / "ti.npy"),
        conditioning={"cells": cells, "facies": observed},
        cell_size=[10.0, 5.0, 2.0],
    )
    variables = [StateVariable(name=f"d{k}", size=600, grid=(30, 20, 1)) for k in range(2)]
    out = prior.generate_joint(variables, 3, np.random.default_rng(9))
    for j in range(3):
        facies = facies_from_distances(np.stack([out["d0"][:, j], out["d1"][:, j]]))
        for (ci, cj, _), code in zip(cells, observed, strict=True):
            assert facies[(ci - 1) + 30 * (cj - 1)] == code


#: A source-built mps_snesim_tree (scikit-mps 0.6.0's ignores soft probabilities).
SOURCE_BUILT = os.environ.get("APEX_MPSLIB_SNESIM")


def _stripes(tmp_path: Path) -> tuple[Path, NDArray[np.int64]]:
    """A training image of stripes 4 cells wide along i, and a 30 x 30 'truth' of them."""
    i = np.arange(60)
    np.save(tmp_path / "ti.npy", np.stack([(i // 4) % 2] * 60, axis=1)[:, :, None])
    return tmp_path / "ti.npy", ((np.arange(30) + 2) // 4) % 2


def _soft_map(tmp_path: Path, truth: NDArray[np.int64], p_true: float, name: str) -> str:
    """Soft probabilities on every cell: ``p_true`` for the truth's facies."""
    p_sand = np.where(truth == 1, p_true, 1 - p_true)[:, None, None]  # (30, 1, 1)
    soft = np.stack(np.broadcast_arrays(1 - p_sand, p_sand), axis=-1) * np.ones((30, 30, 1, 1))
    np.save(tmp_path / name, soft)
    return str(tmp_path / name)


def test_the_bundled_program_ignores_soft_data_and_is_refused(tmp_path: Path) -> None:
    ti, truth = _stripes(tmp_path)
    prior = MpslibSnesim(
        str(ti), template_size=[7, 7, 1],
        soft_probability_file=_soft_map(tmp_path, truth, 0.8, "soft.npy"),
    )  # fmt: skip
    variables = [StateVariable(name=f"d{k}", size=900, grid=(30, 30, 1)) for k in range(2)]
    # scikit-mps 0.6.0's program reads one soft category; a fixed release would pass.
    with pytest.raises(RuntimeError, match="build MPSlib from source"):
        prior.generate_joint(variables, 2, np.random.default_rng(3))


@pytest.mark.skipif(SOURCE_BUILT is None, reason="set APEX_MPSLIB_SNESIM to a source build")
def test_soft_data_steer_the_realisations(tmp_path: Path) -> None:
    """Soft data: MPSlib combines them with the training image's conditional pdf
    (P ∝ P_TI · P_soft; MPSlib ``SNESIM.cpp``). With p = 0.8 for a striped truth's facies
    on every cell, realisations agree with it far more than without (about 0.5, the
    stripes' phase being free), and with the inverted map far less.
    """
    ti, truth = _stripes(tmp_path)
    variables = [StateVariable(name=f"d{k}", size=900, grid=(30, 30, 1)) for k in range(2)]

    def agreement(soft: str | None) -> float:
        prior = MpslibSnesim(
            str(ti), template_size=[7, 7, 1], soft_probability_file=soft,
            program=SOURCE_BUILT,
        )  # fmt: skip
        out = prior.generate_joint(variables, 8, np.random.default_rng(3))
        facies = np.stack(
            [facies_from_distances(np.stack([out["d0"][:, j], out["d1"][:, j]])) for j in range(8)]
        )  # (8, 900)
        return float(np.mean(facies.reshape(8, 30, 30, order="F") == truth[None, :, None]))

    assert agreement(_soft_map(tmp_path, truth, 0.8, "soft.npy")) > 0.7
    assert agreement(_soft_map(tmp_path, truth, 0.2, "inverted.npy")) < 0.3
    assert 0.3 < agreement(None) < 0.7


@pytest.mark.skipif(SOURCE_BUILT is None, reason="set APEX_MPSLIB_SNESIM to a source build")
def test_soft_data_stay_in_their_cell_with_a_fractional_cell_size(tmp_path: Path) -> None:
    """Corners written with %g moved soft data one cell left at dx = 0.3 (review of
    2026-10-04): certain sand in columns i = 4..7 (one stripe of the image) between
    shale columns 3 and 8 must stay there.
    """
    ti, _ = _stripes(tmp_path)
    soft = np.full((30, 30, 1, 2), np.nan)
    soft[3] = [1.0, 0.0]
    soft[4:8] = [0.0, 1.0]
    soft[8] = [1.0, 0.0]
    np.save(tmp_path / "soft.npy", soft)
    variables = [StateVariable(name=f"d{k}", size=900, grid=(30, 30, 1)) for k in range(2)]
    prior = MpslibSnesim(
        str(ti), template_size=[7, 7, 1], cell_size=[0.3, 1.0, 1.0],
        soft_probability_file=str(tmp_path / "soft.npy"), program=SOURCE_BUILT,
    )  # fmt: skip
    out = prior.generate_joint(variables, 4, np.random.default_rng(3))
    for j in range(4):
        grid = facies_from_distances(np.stack([out["d0"][:, j], out["d1"][:, j]]))
        columns = grid.reshape((30, 30), order="F")[3:9].mean(axis=1)
        np.testing.assert_array_equal(columns, [0.0, 1.0, 1.0, 1.0, 1.0, 0.0])


@pytest.mark.skipif(SOURCE_BUILT is None, reason="set APEX_MPSLIB_SNESIM to a source build")
def test_soft_data_contradicting_three_facies_are_refused_not_misread(tmp_path: Path) -> None:
    """With three facies, MPSlib's conflict branch returns a probability as the code
    (``SNESIM.cpp``): certain facies 2 against the image's stripes gave facies 0 at 55 %
    of those cells (review of 2026-10-04). The prior refuses such realisations.
    """
    i = np.arange(60)
    np.save(tmp_path / "ti3.npy", np.stack([(i // 4) % 3] * 60, axis=1)[:, :, None])
    soft = np.full((30, 30, 1, 3), np.nan)
    soft[::5] = [0.0, 0.0, 1.0]
    np.save(tmp_path / "soft.npy", soft)
    variables = [StateVariable(name=f"d{k}", size=900, grid=(30, 30, 1)) for k in range(3)]
    prior = MpslibSnesim(
        str(tmp_path / "ti3.npy"), template_size=[7, 7, 1],
        soft_probability_file=str(tmp_path / "soft.npy"), program=SOURCE_BUILT,
    )  # fmt: skip
    with pytest.raises(RuntimeError, match=r"soft probability 0|non-integer"):
        prior.generate_joint(variables, 8, np.random.default_rng(3))
