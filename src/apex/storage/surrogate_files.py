"""A fitted surrogate's files: a JSON manifest and an ``.npz`` of arrays, no code.

``<name>.json`` is the :class:`~apex.domain.surrogate.SurrogateManifest`: the plugin's
registered name and version, its options, keys, validated domain, gate, the snapshots'
provenance and digest, the fit's seed and stream state, the APEX commit, the plugin's own
manifest, and the arrays file's name and SHA-256. ``<name>.npz`` holds the plugin's arrays
(:meth:`~apex.core.surrogates.SurrogateModel.to_arrays`).

**Nothing in them is executed.** Arrays are read with ``allow_pickle=False`` and must be
numbers, booleans or text; the manifest is JSON, validated strictly. There is no pickle and
no ``torch.load``, under a ``[policy]`` or not.

**Identity.** A surrogate's identity is the SHA-256 of its two files::

    sha256( "apex-surrogate-1" newline || sha256(manifest bytes) || sha256(arrays bytes) )

Both files are written byte-for-byte deterministically (sorted JSON; the ``.npz`` stored
uncompressed, its entries sorted and dated 1980-01-01), so writing the same model twice gives
the same identity, and any changed byte gives another. Reading refuses an arrays file whose
SHA-256 is not the one its manifest records, and, when an identity is expected, files of
another identity.
"""

import hashlib
import io
import json
import math
import os
import re
import zipfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

import numpy as np
from numpy.typing import NDArray
from pydantic import JsonValue, ValidationError

from apex.core.errors import ConfigError, validation_problems
from apex.core.surrogates import SurrogateModel
from apex.domain.surrogate import (
    ArraysReference,
    BuildCost,
    FitRecord,
    GateResult,
    SnapshotProvenance,
    SnapshotReference,
    SurrogateManifest,
    TrainingDomain,
    key_document,
)

#: The arrays file sits beside the manifest, with this suffix.
ARRAYS_SUFFIX = ".npz"

#: Array names: plain words, so that an entry never names a path.
_ARRAY_NAME = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.\-]*")

#: Array kinds a surrogate may store: booleans, integers, floats, complex numbers and text.
_ARRAY_KINDS = frozenset("biufcU")

#: The zip entries' date: fixed, so that the file's bytes depend only on its arrays.
_EPOCH = (1980, 1, 1, 0, 0, 0)


@dataclass(frozen=True)
class StoredSurrogate:
    """A surrogate's files as read (or written): the manifest, the arrays and the identity."""

    path: Path
    manifest: SurrogateManifest
    arrays: Mapping[str, NDArray[np.generic]]
    identity: str


def identity_of(manifest_bytes: bytes, arrays_bytes: bytes) -> str:
    """Return the identity of a surrogate's two files (module docstring)."""
    digest = hashlib.sha256(b"apex-surrogate-1\n")
    digest.update(hashlib.sha256(manifest_bytes).digest())
    digest.update(hashlib.sha256(arrays_bytes).digest())
    return digest.hexdigest()


def _json_value(value: object, where: str) -> JsonValue:
    """Return ``value`` as JSON (tuples as lists), refusing what JSON cannot hold exactly."""
    if value is None or isinstance(value, bool | str):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ConfigError(f"{where} holds a number that is not finite")
        return value
    if isinstance(value, Mapping):
        return _json_object(value, where)
    if isinstance(value, list | tuple):
        return [_json_value(item, f"{where}[{i}]") for i, item in enumerate(value)]
    raise ConfigError(f"{where} holds a {type(value).__name__}, which is not JSON")


def _json_object(value: Mapping[str, object], where: str) -> dict[str, JsonValue]:
    """Return a mapping as a JSON object (:func:`_json_value`), its keys strings."""
    result: dict[str, JsonValue] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise ConfigError(f"{where} has a key that is not a string: {key!r}")
        result[key] = _json_value(item, f"{where}.{key}")
    return result


def _checked_arrays(arrays: Mapping[str, NDArray[np.generic]]) -> dict[str, NDArray[np.generic]]:
    result: dict[str, NDArray[np.generic]] = {}
    for name, value in arrays.items():
        if not _ARRAY_NAME.fullmatch(name):
            raise ConfigError(f"a surrogate's array name must be a plain word, got {name!r}")
        array = np.asarray(value)
        if array.dtype.kind not in _ARRAY_KINDS:
            raise ConfigError(
                f"a surrogate's array {name!r} has dtype {array.dtype}: only numbers, "
                "booleans and text are stored (never objects)"
            )
        # Little-endian always: equal values give equal bytes, so equal identities, whatever
        # byte order the plugin's arrays came in. The shape is kept: ascontiguousarray makes
        # a 0-d array 1-d, and a plugin's scalars (a schema version) must come back as stored.
        little = np.ascontiguousarray(array.astype(array.dtype.newbyteorder("<")))
        result[name] = little.reshape(array.shape)
    return result


def _npz_bytes(arrays: Mapping[str, NDArray[np.generic]]) -> bytes:
    """Return the arrays as an ``.npz`` whose bytes depend only on the arrays."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_STORED) as archive:
        for name in sorted(arrays):
            info = zipfile.ZipInfo(f"{name}.npy", date_time=_EPOCH)
            info.compress_type = zipfile.ZIP_STORED
            # Fixed, so the bytes are the same on every platform (Windows writes 0 by default).
            info.create_system = 3
            info.external_attr = 0o644 << 16
            with archive.open(info, "w", force_zip64=True) as entry:
                np.lib.format.write_array(entry, arrays[name], allow_pickle=False)
    return buffer.getvalue()


def _manifest_bytes(manifest: SurrogateManifest) -> bytes:
    document = manifest.model_dump(mode="json")
    return (json.dumps(document, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def _write_atomically(path: Path, data: bytes) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(data)
    os.replace(temporary, path)


def write_surrogate(
    path: Path,
    *,
    plugin: str,
    model: SurrogateModel,
    domain: TrainingDomain,
    provenance: SnapshotProvenance,
    options: Mapping[str, object] | None = None,
    gate: GateResult | None = None,
    snapshots: SnapshotReference | None = None,
    fit: FitRecord | None = None,
    apex_commit: str | None = None,
) -> StoredSurrogate:
    """Write a fitted surrogate as ``path`` (a ``.json`` manifest) and its ``.npz`` beside it.

    The arrays are written first and the manifest last, each atomically, so a manifest
    always names arrays that exist.

    Raises
    ------
    ConfigError
        If ``path`` is not a ``.json`` file, the plugin's manifest or options are not JSON
        (or hold numbers that are not finite), an array holds objects, or the domain names
        keys the model does not answer.
    """
    path = Path(path)
    if path.suffix != ".json":
        raise ConfigError(f"a surrogate's manifest must be a .json file, got {path.name!r}")
    own, raw_arrays = model.to_arrays()
    keys = tuple(model.keys)
    missing = set(domain.keys) - set(keys)
    if missing:
        raise ConfigError(
            f"the domain names {len(missing)} key(s) the model does not answer, e.g. "
            f"{sorted(missing)[0]}"
        )
    arrays = _checked_arrays(raw_arrays)
    arrays_bytes = _npz_bytes(arrays)
    model_document = _json_object(own, "the plugin's manifest")
    options_document = _json_object(options or {}, "the plugin's options")
    version = getattr(model, "version", None)
    manifest = SurrogateManifest(
        plugin=plugin,
        plugin_version=version if isinstance(version, str) else None,
        options=options_document,
        keys=[key_document(k) for k in keys],
        fields=list(domain.fields),
        field_times=list(domain.field_times),
        domain=domain.to_document(),
        gate=gate,
        provenance=provenance,
        snapshots=snapshots,
        fit=fit or FitRecord(),
        apex_commit=apex_commit,
        model=model_document,
        arrays=ArraysReference(
            file=path.with_suffix(ARRAYS_SUFFIX).name,
            sha256=hashlib.sha256(arrays_bytes).hexdigest(),
        ),
    )
    manifest_bytes = _manifest_bytes(manifest)
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_atomically(path.with_suffix(ARRAYS_SUFFIX), arrays_bytes)
    _write_atomically(path, manifest_bytes)
    return StoredSurrogate(
        path=path,
        manifest=manifest,
        arrays=MappingProxyType(arrays),
        identity=identity_of(manifest_bytes, arrays_bytes),
    )


def cost_path(manifest: Path) -> Path:
    """Return the file of a model's build cost: ``<manifest stem>.cost.json`` beside it."""
    return Path(manifest).with_name(f"{Path(manifest).stem}.cost.json")


def write_cost(manifest: Path, cost: BuildCost) -> Path:
    """Write a model's build cost beside its manifest (outside its identity); return the file."""
    path = cost_path(manifest)
    _write_atomically(path, (cost.model_dump_json(indent=2) + "\n").encode())
    return path


def read_cost(manifest: Path) -> BuildCost | None:
    """Return a model's build cost, ``None`` when it has none (a refit, an older build).

    Raises
    ------
    ConfigError
        If the file is there but is not a build cost.
    """
    path = cost_path(manifest)
    if not path.is_file():
        return None
    try:
        return BuildCost.model_validate_json(path.read_bytes())
    except ValueError as error:
        raise ConfigError(f"{path} is not a surrogate's build cost: {error}") from error


def read_surrogate(path: Path, *, identity: str | None = None) -> StoredSurrogate:
    """Read a surrogate's manifest and arrays, executing nothing.

    Parameters
    ----------
    path
        The manifest (``.json``).
    identity
        The identity expected (a run's recorded one); ``None`` accepts any.

    Raises
    ------
    ConfigError
        If a file is missing or not what it should be, the arrays file is not the one the
        manifest records, an array holds objects, or the identity differs.
    """
    path = Path(path)
    try:
        manifest_bytes = path.read_bytes()
    except OSError as error:
        raise ConfigError(f"cannot read the surrogate manifest {path}: {error}") from None
    try:
        manifest = SurrogateManifest.model_validate(json.loads(manifest_bytes))
    except json.JSONDecodeError as error:
        raise ConfigError(f"the surrogate manifest {path.name} is not JSON: {error}") from None
    except ValidationError as error:
        raise ConfigError(
            f"the surrogate manifest {path.name} is not valid:\n{validation_problems(error)}"
        ) from None
    name = manifest.arrays.file
    # A bare file name beside the manifest: a manifest must not make APEX read anything else.
    if Path(name).name != name or name in ("", ".", "..") or not name.endswith(ARRAYS_SUFFIX):
        raise ConfigError(
            f"the surrogate manifest {path.name} names arrays {name!r}: only a .npz file "
            "beside it is read"
        )
    try:
        arrays_bytes = (path.parent / name).read_bytes()
    except OSError as error:
        raise ConfigError(f"cannot read the surrogate's arrays {name}: {error}") from None
    if hashlib.sha256(arrays_bytes).hexdigest() != manifest.arrays.sha256:
        raise ConfigError(
            f"the surrogate's arrays {name} are not the ones its manifest {path.name} records "
            "(the file changed since the model was written)"
        )
    found = identity_of(manifest_bytes, arrays_bytes)
    if identity is not None and found != identity:
        raise ConfigError(
            f"the surrogate {path.name} has identity {found[:12]}..., not the "
            f"{identity[:12]}... expected: the model file changed"
        )
    try:
        with np.load(io.BytesIO(arrays_bytes), allow_pickle=False) as archive:
            arrays = {key: archive[key] for key in archive.files}
    except (ValueError, OSError, zipfile.BadZipFile) as error:
        raise ConfigError(f"cannot read the surrogate's arrays {name}: {error}") from None
    return StoredSurrogate(
        path=path,
        manifest=manifest,
        arrays=MappingProxyType(_checked_arrays(arrays)),
        identity=found,
    )
