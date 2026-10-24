"""Prior generator that reads a stored ensemble."""

from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from apex.core.options import InputFile, Text
from apex.core.plugins import priors
from apex.domain.state import StateVariable


@priors.register("file")
class FilePrior:
    """Read a stored ensemble ``(size, n_ensemble)`` from an ``.npz`` file.

    Parameters
    ----------
    ensemble_file
        ``.npz`` with one array per variable.
    key
        Array name; defaults to the variable's name.

    With ``[prior] ensemble_size`` set, the first ``ensemble_size`` columns are used;
    otherwise all of them.
    """

    #: A stored ensemble is used as is, not clipped to the state bounds (as PET's
    #: ``importstate``).
    clip = False

    def __init__(self, ensemble_file: InputFile, key: Text | None = None) -> None:
        self.path = Path(ensemble_file)
        if not self.path.is_file():
            raise ValueError(f"prior ensemble file not found: {self.path}")
        self.key = key

    def generate(
        self, variable: StateVariable, n_ensemble: int | None, rng: np.random.Generator
    ) -> NDArray[np.float64]:
        """Return the stored ensemble of ``variable``."""
        key = self.key or variable.name
        with np.load(self.path) as data:
            if key not in data:
                raise ValueError(f"{self.path} has no array {key!r} (has {data.files})")
            block = np.asarray(data[key], dtype=np.float64)
        if block.ndim != 2 or block.shape[0] != variable.size:
            raise ValueError(
                f"{self.path}[{key!r}] must be ({variable.size}, n_ensemble), got {block.shape}"
            )
        if n_ensemble is None:
            return block
        if block.shape[1] < n_ensemble:
            raise ValueError(
                f"{self.path}[{key!r}] has {block.shape[1]} members, need {n_ensemble}"
            )
        return block[:, :n_ensemble]


def read_array(path: Path, key: str | None = None) -> NDArray[np.float64]:
    """Read one array from ``.npz`` (``key``, or its only/first array) or ``.npy``.

    Raises
    ------
    ValueError
        If the file is missing or the key is absent.
    """
    if not path.is_file():
        raise ValueError(f"file not found: {path}")
    if path.suffix == ".npy":
        return np.asarray(np.load(path), dtype=np.float64)
    with np.load(path) as data:
        name = key or data.files[0]
        if name not in data:
            raise ValueError(f"{path} has no array {name!r} (has {data.files})")
        return np.asarray(data[name], dtype=np.float64)
