"""Parity tests compare APEX methods against stored PET reference outputs.

PET runs in its own environment. Reference files are produced by
``tools/pet_reference/generate.py`` running under ``.venv-pet``, and stored in
``tests/data/reference/pet/``. Mark every test here with ``@pytest.mark.pet_parity``.
"""

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

PET_REFERENCE_DIR = Path(__file__).resolve().parents[1] / "data" / "reference" / "pet"

type ReferenceLoader = Callable[[str], dict[str, NDArray[np.generic]]]


@pytest.fixture
def pet_reference() -> ReferenceLoader:
    """Return a loader for ``tests/data/reference/pet/<case>.npz``.

    Keys prefixed ``in_`` are the seeded inputs, ``out_`` are PET's outputs and
    ``meta_`` record provenance (PET commit, numpy version, seed).
    """

    def load(case: str) -> dict[str, NDArray[np.generic]]:
        path = PET_REFERENCE_DIR / f"{case}.npz"
        if not path.exists():
            # Fail rather than skip: a missing reference silently disables parity checking.
            pytest.fail(
                f"missing PET reference {path}; generate it with "
                f"`env -u PYTHONPATH .venv-pet/bin/python tools/pet_reference/generate.py {case}`"
            )
        with np.load(path) as data:
            return {key: data[key] for key in data.files}

    return load
