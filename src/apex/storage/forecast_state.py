"""A prediction's forecast-state file for a model-error model that extrapolates.

The model's fitted arrays (:meth:`apex.core.simulation.ExtrapolatingModel.forecast_state`)
and the prediction's seed (:data:`apex.core.simulation.SEED_ENTROPY`,
:data:`~apex.core.simulation.SEED_SPAWN_KEY`) in one ``.npz`` whose bytes depend only on
them: sorted entries, a fixed time stamp, no compression, little-endian arrays, no pickles.
The prediction names it by a digest of the fit's inputs (``forecast_state-<digest>.npz``):
a resume, or a configuration replacing an unfinished prediction, with the same inputs finds
the file and reuses it rather than fitting again.
"""

import hashlib
import io
import os
import zipfile
from collections.abc import Mapping
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from apex.core.simulation import SEED_ENTROPY, SEED_SPAWN_KEY

__all__ = ["forecast_state_bytes", "write_forecast_state"]

_EPOCH = (1980, 1, 1, 0, 0, 0)
_KINDS = frozenset("biufU")


def forecast_state_bytes(
    arrays: Mapping[str, NDArray[np.generic]], seed: np.random.SeedSequence
) -> bytes:
    """Return the file's bytes: the arrays and the seed, deterministically.

    Raises
    ------
    ValueError
        If an array is not numbers, booleans or text, or is named as the seed's arrays.
    """
    clash = sorted({SEED_ENTROPY, SEED_SPAWN_KEY} & set(arrays))
    if clash:
        raise ValueError(f"a forecast state may not name its arrays {clash} (the seed's)")
    entries: dict[str, NDArray[np.generic]] = {
        **arrays,
        SEED_ENTROPY: np.array(str(seed.entropy)),
        SEED_SPAWN_KEY: np.array(seed.spawn_key, dtype=np.int64),
    }
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_STORED) as archive:
        for name in sorted(entries):
            array = np.asarray(entries[name])
            if array.dtype.kind not in _KINDS:
                raise ValueError(
                    f"a forecast state's array {name!r} has dtype {array.dtype}: only numbers, "
                    "booleans and text are stored"
                )
            little = np.ascontiguousarray(array.astype(array.dtype.newbyteorder("<")))
            info = zipfile.ZipInfo(f"{name}.npy", date_time=_EPOCH)
            info.compress_type = zipfile.ZIP_STORED
            info.create_system = 3
            info.external_attr = 0o644 << 16
            with archive.open(info, "w", force_zip64=True) as entry:
                np.lib.format.write_array(entry, little.reshape(array.shape), allow_pickle=False)
    return buffer.getvalue()


def write_forecast_state(
    path: Path, arrays: Mapping[str, NDArray[np.generic]], seed: np.random.SeedSequence
) -> str:
    """Write the file at ``path`` (its folder made if missing); return its SHA-256.

    Written to ``<path>.tmp`` and renamed (a name a prediction's folder counts as its own,
    so a crash's leftover does not block replacing it).
    """
    data = forecast_state_bytes(arrays, seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(data)
    os.replace(temporary, path)
    return hashlib.sha256(data).hexdigest()
