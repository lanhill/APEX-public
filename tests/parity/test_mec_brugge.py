"""PG-LSTM features and residuals against PET-fork on its Brugge run.

Opt-in twice (``-m integration`` and ``APEX_MEC_BRUGGE``), since it reads PET-fork's
Eclipse run of the Brugge reference model, which is not in this repository:

    APEX_MEC_BRUGGE=/path/to/PET-fork/Examples/.../Brugge \\
        uv run pytest -m integration tests/parity/test_mec_brugge.py

The reference ``mec_brugge_features`` holds PET-fork's input sequences and residuals of
all 30 wells for seeded parameters (``tools/pet_reference/generate.py``). APEX reads the
same schedule and run itself, so this checks the schedule walk, the block union and its
order, every feature (INIT, restart, the report-step WBHP) and the networks end to end.
"""

import os
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from apex.algorithms.model_error.lstm import well_residuals
from apex.simulation.model_error import CaseFiles
from apex.workflows.model_error import PGLSTMModel

type ReferenceLoader = Callable[[str], dict[str, NDArray[np.generic]]]

BRUGGE = os.environ.get("APEX_MEC_BRUGGE")
REPORT_STEPS = [1, 2, 3, 4, 5, 6, 10, 20, 23, 24, 25, 26, 27, 35, 36, 60, 80, 100, 126, 127]

pytestmark = [
    pytest.mark.integration,
    pytest.mark.pet_parity,
    pytest.mark.skipif(not BRUGGE, reason="set APEX_MEC_BRUGGE to PET-fork's Brugge folder"),
]


def test_features_and_residuals_match_pet_fork(pet_reference: ReferenceLoader) -> None:
    assert BRUGGE is not None
    folder = Path(BRUGGE)
    ref = pet_reference("mec_brugge_features")
    producers = [f"BR-P-{i}" for i in range(1, 21)]
    injectors = [f"BR-I-{i}" for i in range(1, 11)]
    model = PGLSTMModel(
        folder / "include" / "Hist", REPORT_STEPS, producers, injectors, shut_wells="ignore"
    )
    assert model.n_params == 352_920  # the manuscript's count for these 30 networks

    features = model.features(CaseFiles(folder / "model_error" / "BRUGGE_REF_FAULTS"))

    for well, x, network in zip(model.wells, features, model.networks, strict=True):
        name = well.name
        assert well.blocks == tuple(tuple(int(v) for v in b) for b in ref[f"out_{name}_blocks"])
        expected = np.asarray(ref[f"out_{name}_features"], dtype=np.float64)
        # PET-fork hands torch float32 inputs; APEX reads the same float32 files.
        np.testing.assert_allclose(x, expected, rtol=1e-6, atol=1e-6)
        scale = np.asarray(ref[f"in_{name}_scale"], dtype=np.float64)
        theta = np.asarray(ref[f"in_{name}_theta"], dtype=np.float64)
        residuals = well_residuals(
            network, x[:, :, None], theta[:, None], scale, np.ones((20, len(well.outputs)), bool)
        )[:, :, 0]
        expected_r = np.asarray(ref[f"out_{name}_residuals"], dtype=np.float64)
        # Float32 arithmetic in different orders (as tests/parity/test_mec.py), relative to
        # each output's own scale (WWCT's is ~0.07, WOPR's ~2000): measured 6e-7.
        assert (np.abs(residuals - expected_r) / scale[None, :]).max() <= 1e-5
        # The well status, which the default shut_wells = "zero" masks with: PET-fork
        # stores it per output, with WBHP always open.
        expected_open = np.asarray(ref[f"out_{name}_open"], dtype=bool)
        for k, output in enumerate(well.outputs):
            assert np.array_equal(
                expected_open[:, k], np.ones(20, bool) if output == "WBHP" else well.open
            ), (name, output)
