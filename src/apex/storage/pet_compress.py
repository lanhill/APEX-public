"""Convert PET's ``[dataassim] compress`` block into APEX ``wavelet`` data transforms.

PET compresses *vintages*: it walks the observation table row by row, then column by
column, and every non-empty cell of a ``compress_data`` type is the next vintage, with the
next ``mask`` file and ``min_noise`` entry (``DataReader.get_data``). Cells beyond the
masks given stay uncompressed. This module repeats that walk, then groups the vintages of
one data type that share a mask and ``min_noise`` into one transform with their ``times``.

What PET runs is reproduced where APEX can, and noted (see ``docs/migration/pet_parity.md``,
D1-D6):

- the mask is written as the flat vector PET applies (C-order flattening, D2);
- a missing mask file means all cells are active, as in PET (D5);
- ``threshold_rule = "bayesian"`` with white noise becomes ``"universal"``, which is what
  PET computes (D4);
- soft thresholding cannot be reproduced exactly: APEX also shrinks the predictions (D1).

``use_ensemble`` is refused, as PET refuses it.
"""

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from apex.core.errors import ConfigError
from apex.storage.pet_format import PetField, PetScalar

#: ``compress`` keys PET reads with ``info[key]`` (so it fails without them).
REQUIRED = (
    "compress_data",
    "dim",
    "mask",
    "level",
    "wname",
    "threshold_rule",
    "th_mult",
    "order",
    "min_noise",
    "inactive_value",
)

#: ``compress`` keys PET reads with a default.
OPTIONAL = ("colored_noise", "use_hard_th", "keep_ca", "use_ensemble")


@dataclass
class Compression:
    """The transforms a ``compress`` block becomes, and which fields they compress."""

    transforms: list[dict[str, Any]] = field(default_factory=list)
    compressed: set[tuple[str, str]] = field(default_factory=set)
    notes: list[str] = field(default_factory=list)


@dataclass
class _Group:
    datatype: str
    mask: Path | None
    min_noise: float
    times: list[str] = field(default_factory=list)


def convert_compress(
    block: Mapping[str, Any],
    entries: Sequence[PetScalar | PetField],
    base: Path,
    output_dir: Path,
) -> Compression:
    """Map a PET ``compress`` block to ``[[data_transforms]]`` sections.

    Parameters
    ----------
    block
        The ``compress`` table, keys lower-cased.
    entries
        The observation cells in PET's order (:func:`~apex.storage.pet_format.read_pet_data`).
    base
        Directory PET's relative mask paths are resolved against.
    output_dir
        The APEX case; mask files are written to ``output_dir/masks``.

    Raises
    ------
    ConfigError
        If a key is unknown, missing or invalid, ``use_ensemble`` is on, a compressed type
        has scalar cells or none at all, or a mask does not fit the grid.
    """
    info = {key: _flag(value) for key, value in block.items()}
    unknown = sorted(set(info) - set(REQUIRED) - set(OPTIONAL))
    if unknown:
        raise ConfigError(f"[dataassim.compress]: unknown keys {', '.join(unknown)}")
    missing = [key for key in REQUIRED if key not in info]
    if missing:
        raise ConfigError(f"[dataassim.compress]: PET requires {', '.join(missing)}")
    if info.get("use_ensemble"):
        raise ConfigError(
            "[dataassim.compress]: use_ensemble is not supported (PET refuses it as well)"
        )

    for key in ("colored_noise", "use_hard_th", "keep_ca"):
        if key in info and not isinstance(info[key], bool):
            raise ConfigError(
                f"[dataassim.compress]: {key} must be yes/no or a boolean, got {info[key]!r}"
            )
    types = _strings(info["compress_data"], "compress_data")
    if isinstance(info["compress_data"], str):
        # PET's reader tests `column in compress_data`, a substring test for a string.
        similar = sorted(
            {e.datatype for e in entries if e.datatype != info["compress_data"]}
            & {e.datatype for e in entries if e.datatype in info["compress_data"]}
        )
        if similar:
            raise ConfigError(
                f"[dataassim.compress]: PET would also compress {', '.join(similar)} "
                f"(substrings of compress_data = {info['compress_data']!r}); give "
                "compress_data as a list"
            )
    result = Compression()
    masks = _mask_sources(_strings(info["mask"], "mask"), base, result.notes)
    min_noise = info["min_noise"]
    grid = [int(n) for n in _list(info["dim"], "dim")]
    order = str(info["order"]).upper()

    # PET's walk: one vintage per non-empty cell of a compressed type, in table order.
    groups: dict[tuple[str, Path | None, float], _Group] = {}  # (type, mask, min_noise)
    vintage = 0
    beyond: list[str] = []
    for entry in entries:
        if entry.datatype not in types:
            continue
        where = f"{entry.datatype}@{entry.time}"
        if vintage >= len(masks):
            beyond.append(where)
            vintage += 1
            continue
        if isinstance(entry, PetScalar):
            # PET would broadcast it into a constant field and use up a vintage.
            raise ConfigError(f"[dataassim.compress]: {where} is a scalar; only fields compress")
        noise = _vintage_noise(min_noise, vintage)
        mask = masks[vintage]
        group = groups.setdefault(
            (entry.datatype, mask, noise), _Group(entry.datatype, mask, noise)
        )
        group.times.append(entry.time)
        result.compressed.add((entry.time, entry.datatype))
        vintage += 1
    if not groups:
        raise ConfigError(f"[dataassim.compress]: no field data of {', '.join(types)} observed")
    if beyond:
        result.notes.append(
            f"compress: {', '.join(beyond)} lie beyond the {len(masks)} mask(s) given and stay "
            "uncompressed, as in PET"
        )

    common = _common_options(info, grid, order, result.notes)
    written: dict[Path, str] = {}
    for group in groups.values():
        options: dict[str, Any] = {"datatype": group.datatype, **common}
        options["min_noise"] = group.min_noise
        options["times"] = group.times
        if group.mask is not None:
            if group.mask not in written:
                written[group.mask] = _write_mask(
                    group.mask, grid, order, output_dir, len(written), result
                )
            options["mask_file"] = written[group.mask]
        result.transforms.append({"name": "wavelet", "options": options})
    return result


def _common_options(
    info: Mapping[str, Any], grid: list[int], order: str, notes: list[str]
) -> dict[str, Any]:
    if order not in ("C", "F"):
        raise ConfigError(f"[dataassim.compress]: order must be 'C' or 'F', got {order!r}")
    try:
        multiplier = float(info["th_mult"])
        level = int(info["level"])
        inactive = float(info["inactive_value"])
    except (TypeError, ValueError) as error:
        raise ConfigError(f"[dataassim.compress]: {error}") from error
    if not math.isfinite(multiplier) or multiplier < 0:
        # PET does not threshold at all then, and later fails without leading indices.
        raise ConfigError(f"[dataassim.compress]: th_mult must be >= 0, got {multiplier}")
    noise = "coloured" if info.get("colored_noise", False) else "white"
    rule = str(info["threshold_rule"])
    if noise == "white":
        if rule != "universal":
            notes.append(
                f"compress: threshold_rule = {rule!r} with white noise; PET uses the "
                "universal rule there (D4), and so does the converted case"
            )
        rule = "universal"
    elif rule not in ("universal", "bayesian"):
        # PET compares case-sensitively and raises for anything else.
        raise ConfigError(f"[dataassim.compress]: unknown threshold_rule {rule!r}")
    hard = bool(info.get("use_hard_th", False))
    if not hard:
        notes.append(
            "compress: soft thresholding; APEX also shrinks the predicted coefficients by the "
            "observed thresholds, PET does not (D1), so results differ from PET's"
        )
    return {
        "grid": grid,
        "wavelet": str(info["wname"]),
        "level": level,
        "order": order,
        "noise": noise,
        "threshold_rule": rule,
        "threshold_multiplier": multiplier,
        "thresholding": "hard" if hard else "soft",
        "keep_approximation": bool(info.get("keep_ca", False)),
        "inactive_value": inactive,
    }


def _mask_sources(names: list[str], base: Path, notes: list[str]) -> list[Path | None]:
    """Resolve PET's mask files as it reads them; ``None`` means all cells active.

    PET's ``organize_sparse_representation`` replaces the k-th (1-based) missing file by an
    all-true mask written to ``mask_k.npz`` in its working directory (taken to be the
    configuration's directory), so a later entry of that name reads all-true too (D5).
    """
    sources: list[Path | None] = []
    overwritten: set[Path] = set()
    for k, name in enumerate(names, start=1):
        path = (base / name).resolve()
        if not path.is_file():
            notes.append(f"compress: mask file {path} not found; all cells active, as PET assumes")
            sources.append(None)
            overwritten.add((base / f"mask_{k}.npz").resolve())
        elif path in overwritten:
            notes.append(
                f"compress: PET overwrites {path} with an all-true mask before reading it, "
                "because an earlier mask file is missing (D5); reproduced as all active"
            )
            sources.append(None)
        else:
            sources.append(path)
    return sources


def _write_mask(
    source: Path, grid: list[int], order: str, output_dir: Path, index: int, result: Compression
) -> str:
    """Write the flat mask PET applies; return its path relative to the case."""
    with np.load(source) as data:
        if "mask" not in data:
            raise ConfigError(f"{source}: PET mask files hold an array named 'mask'")
        mask = np.asarray(data["mask"])
    n_cells = math.prod(grid)
    if mask.size != n_cells:
        raise ConfigError(f"{source}: mask has {mask.size} cells, dim {grid} has {n_cells}")
    # PET flattens the mask in C order and indexes the data (in `order`) with it (D2).
    if not np.isin(mask, (0, 1)).all():
        raise ConfigError(f"{source}: a mask must be boolean (0/1)")
    applied = mask.reshape(-1).astype(bool)
    if mask.ndim > 1 and order == "F" and not np.array_equal(applied, mask.reshape(-1, order="F")):
        result.notes.append(
            f"compress: {source.name} is applied as PET does, flattened in C order although "
            "the data are in F order (D2); to mask the intended cells, write the mask flat "
            "in F order"
        )
    target = output_dir / "masks" / f"mask_{index}.npz"
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez(target, mask=applied)
    return str(target.relative_to(output_dir))


def _vintage_noise(min_noise: object, vintage: int) -> float:
    if isinstance(min_noise, list):
        if vintage >= len(min_noise):
            raise ConfigError(
                "[dataassim.compress]: min_noise must be a scalar or one number per vintage "
                f"(vintage {vintage + 1} has none)"
            )
        return float(min_noise[vintage])
    return float(min_noise)  # type: ignore[arg-type]


def _flag(value: object) -> object:
    # PET turns 'yes'/'no' strings into booleans.
    if value == "yes":
        return True
    if value == "no":
        return False
    return value


def _strings(value: object, key: str) -> list[str]:
    return [value] if isinstance(value, str) else [str(v) for v in _list(value, key)]


def _list(value: object, key: str) -> list[Any]:
    if not isinstance(value, list) or not value:
        raise ConfigError(f"[dataassim.compress]: {key} must be a non-empty list, got {value!r}")
    return value
