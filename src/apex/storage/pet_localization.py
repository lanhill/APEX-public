"""Convert PET's ``[dataassim.localization]`` block into an APEX localization.

Upstream PET (``e65fb10``, ``pipt/localization/factory.py``) has two modes, both for the
approx analysis of ES, ES-MDA, LM-EnRML and GN-EnRML:

- ``name = "autoadaloc"`` (``auto_ada_loc.py``) with ``projection = "rank-r"`` (the
  default) is ``adaptive_projected``: ``threshold`` and ``cutoff`` (also ``nstd``, or the
  value of the ``autoadaloc`` key) carry over, ``type`` becomes ``taper`` (``sigm`` is
  ``sigmoid``). ``projection = "ensemble"`` cannot run in PET and is refused. PET's noise
  levels are per ``parameters`` block; APEX's are per state variable, which the importer's
  state variables are (the ``field`` and ``parameters`` keys are therefore not needed).
- ``name = "distance_loc"`` (``distance_localization.py``) is ``distance_entries``. PET's
  ``field`` ``[nz, ny, nx]`` becomes APEX's ``grid`` ``[nx, ny, nz]``, and each entry's
  0-based ``x, y, z`` along that field (PET's ``x`` is Eclipse ``j``, its ``y`` is ``i``)
  becomes the 1-based Eclipse ``position`` ``[y + 1, x + 1, z + 1]``. Entries are inline
  tables, row strings, or a CSV of rows (PET's ``entries``). Kernels ``gc``, ``fb``,
  ``region`` and ``import`` (its ``file`` copied into the case) map to APEX's names,
  ``z_range = ":"`` to ``"all"``, ``aniso`` and ``rotation`` to ``anisotropy``. ``actnum``
  (a ``.npz``) is copied into the case as ``actnum_file``. PET ignores an entry naming data,
  a time or a variable the case lacks; such entries are left out and noted (APEX would
  refuse them).

Unknown keys fail loudly, so that nothing changes the update silently.
"""

import csv
import math
import shutil
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from apex.core.errors import ConfigError

#: Keys each PET mode reads (``name`` included).
AUTOADALOC_KEYS = frozenset({
    "name", "field", "actnum", "parameters", "threshold", "cutoff", "nstd", "autoadaloc",
    "type", "projection",
})  # fmt: skip
DISTANCE_KEYS = frozenset({"name", "field", "actnum", "parameters", "entries", "taper_func"})
#: PET's taper ``type`` -> APEX's ``taper``.
TAPER_TYPES = {"hard": "hard", "soft": "soft", "sigm": "sigmoid"}
#: PET's kernels -> APEX's.
KERNELS = {"gc": "gaspari_cohn", "fb": "furrer_bengtsson", "region": "region", "import": "import"}


def convert_localization(
    info: object,
    *,
    strategy: str,
    datatypes: Sequence[str],
    times: Sequence[str],
    variables: Sequence[str],
    base: Path,
    output_dir: Path,
    notes: list[str],
    time_labels: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Return the APEX ``[method.localization]`` section for PET's localization block.

    Parameters
    ----------
    info
        PET's ``[dataassim.localization]`` table.
    strategy
        The converted method's update strategy (PET localizes the approx analysis only).
    datatypes, times, variables
        The case's data types and report times (PET's column names and row labels, as the
        importer keeps them) and state variables, to resolve the entries' names.
    base, output_dir
        PET's configuration folder (relative files) and the APEX case (copied files).
    notes
        Receives what the conversion left out.
    time_labels
        PET's report labels and the case's times they became (days after the start
        date, as ISO dates), so entries naming PET's labels resolve.

    Raises
    ------
    ConfigError
        For a mode, key or entry APEX cannot reproduce.
    """
    if not isinstance(info, Mapping):
        raise ConfigError(
            f"PET localization must be a table ([dataassim.localization] with a name); got {info!r}"
        )
    table = {str(k).lower(): v for k, v in info.items()}
    name = localization_name(table)
    if strategy != "approx":
        raise ConfigError(
            f"PET localizes the approx analysis only; this case's analysis is {strategy!r}"
        )
    if name == "autoadaloc":
        return _autoadaloc(table, notes)
    if name == "distance_loc":
        return _distance(
            table, datatypes, times, variables, base, output_dir, notes, time_labels or {}
        )
    raise ConfigError(
        f"PET localization {name!r} is not ported (supported: autoadaloc, distance_loc; "
        "PET itself refuses localanalysis and parallel_update)"
    )


def localization_name(table: Mapping[str, object]) -> str:
    """Return the PET localization mode a table names, or infers from its keys."""
    return str(table.get("name") or _inferred_name(table)).lower()


def _inferred_name(table: Mapping[str, object]) -> str:
    # PET's normalize_parsed_info names the mode from its keys when no name is given.
    if "autoadaloc" in table:
        return "autoadaloc"
    if "entries" in table or any(str(k).endswith(".csv") for k in table):
        return "distance_loc"
    raise ConfigError("PET localization has no name (autoadaloc or distance_loc)")


def _unknown(table: Mapping[str, object], known: frozenset[str], mode: str) -> None:
    unknown = sorted(set(table) - known)
    if unknown:
        raise ConfigError(f"PET localization {mode}: unknown keys {', '.join(unknown)}")


def _autoadaloc(table: Mapping[str, object], notes: list[str]) -> dict[str, Any]:
    _unknown(table, AUTOADALOC_KEYS, "autoadaloc")
    projection = str(table.get("projection", "rank-r")).lower()
    if projection != "rank-r":
        raise ConfigError(
            f"PET autoadaloc projection {projection!r} is not ported: 'ensemble' fails in "
            "PET unless r == N == n_data (docs/migration/pet_parity.md); APEX's "
            "adaptive_per_update tapers in the ensemble space"
        )
    taper = str(table.get("type", "hard")).lower()
    if taper not in TAPER_TYPES:
        raise ConfigError(f"PET autoadaloc type {taper!r}: expected hard, soft or sigm")
    options: dict[str, Any] = {
        "threshold": str(table.get("threshold", "adaptive")).lower(),
        "cutoff": _cutoff(table),
        "taper": TAPER_TYPES[taper],
    }
    if "parameters" in table or "actnum" in table:
        notes.append(
            "localization: autoadaloc's noise levels are per state variable in APEX "
            "(PET: per `parameters` block of `field` or active cells)"
        )
    return {"name": "adaptive_projected", "options": options}


def _cutoff(table: Mapping[str, object]) -> float:
    # As PET's _cutoff_from: cutoff, nstd, or the value of the autoadaloc key itself
    # (`autoadaloc = true` is a flag, not a value).
    for key in ("cutoff", "nstd", "autoadaloc"):
        value = table.get(key)
        if value is None or isinstance(value, bool):
            continue
        try:
            return float(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            continue
    return 0.3


def _distance(
    table: Mapping[str, object],
    datatypes: Sequence[str],
    times: Sequence[str],
    variables: Sequence[str],
    base: Path,
    output_dir: Path,
    notes: list[str],
    time_labels: Mapping[str, str],
) -> dict[str, Any]:
    csv_keys = [k for k in table if str(k).endswith(".csv")]
    _unknown({k: v for k, v in table.items() if k not in csv_keys}, DISTANCE_KEYS, "distance_loc")
    field = table.get("field")
    if not isinstance(field, list) or len(field) != 3:
        raise ConfigError("PET distance_loc needs `field = [nz, ny, nx]`")
    nz, ny, nx = (int(n) for n in field)
    rows: list[str | Mapping[str, object]] = []
    entries = table.get("entries")
    if isinstance(entries, str):
        rows.extend(_csv_rows(_path(entries, base)))
    elif isinstance(entries, list):
        rows.extend(entries)
    elif entries is None and csv_keys:
        rows.extend(_csv_rows(_path(csv_keys[0], base)))
    else:
        raise ConfigError("PET distance_loc needs `entries` (a list, or a CSV file)")
    options: dict[str, Any] = {"grid": [nx, ny, nz], "entries": []}
    resolve = _Names(datatypes, times, variables, time_labels)
    for n, row in enumerate(rows):
        entry = _entry(row, n, base, output_dir)
        named = resolve(entry)
        if named is None:
            notes.append(
                f"localization: distance_loc entry {n} names data, a time or a variable the "
                "case lacks; PET ignores it, so it is left out"
            )
            continue
        options["entries"].append(named)
    if not options["entries"]:
        raise ConfigError("PET distance_loc: no entry names this case's data and state")
    if "actnum" in table:
        source = _path(str(table["actnum"]), base)
        if source.suffix != ".npz":
            raise ConfigError(f"PET distance_loc actnum must be a .npz file: {source}")
        shutil.copy2(source, output_dir / "localization_actnum.npz")
        options["actnum_file"] = "localization_actnum.npz"
    if "parameters" in table:
        notes.append("localization: distance_loc applies to the case's state variables")
    return {"name": "distance_entries", "options": options}


def _path(name: str, base: Path) -> Path:
    path = Path(name) if Path(name).is_absolute() else base / name
    if not path.is_file():
        raise ConfigError(f"PET localization file not found: {path}")
    return path


def _csv_rows(path: Path) -> list[str | Mapping[str, object]]:
    # PET reads every cell of the CSV as one row string.
    with path.open(newline="") as handle:
        return [cell for line in csv.reader(handle) for cell in line if cell.strip()]


def _entry(row: str | Mapping[str, object], n: int, base: Path, output_dir: Path) -> dict[str, Any]:
    """Return one entry in APEX's form, its names still PET's."""
    if isinstance(row, str):
        row = _row_table(row, n)
    item: dict[str, Any] = {str(k).lower(): v for k, v in row.items()}
    kernel = str(item.get("taper", "")).lower()
    if kernel not in KERNELS:
        raise ConfigError(f"PET distance_loc entry {n}: taper {kernel!r} (gc, fb, region, import)")
    entry: dict[str, Any] = {
        "data": str(item.get("data_type", "*")),
        "time": str(item.get("time", "*")),
        "parameter": str(item.get("param", "*")),
        "taper": KERNELS[kernel],
        "z_range": _z_range(item.get("z_range", ":"), n),
    }
    if kernel == "import":
        source = _path(str(item.get("file", "")), base)
        target = f"localization_taper_{n}{source.suffix}"
        shutil.copy2(source, output_dir / target)
        entry["taper_file"] = target
        return entry
    # PET's x runs along its field's ny axis (Eclipse j) and its y along nx (Eclipse i).
    try:
        x, y, z = int(item["x"]), int(item["y"]), int(item.get("z", 0))
        entry["position"] = [y + 1, x + 1, z + 1]
        if kernel != "region":
            entry["radius"] = float(int(item["radius"]))  # PET truncates the radius
        entry["anisotropy"] = [float(item.get("aniso", 1.0)), float(item.get("rotation", 0.0))]
    except (KeyError, TypeError, ValueError) as error:
        raise ConfigError(f"PET distance_loc entry {n}: {error!r} in {dict(item)}") from error
    return entry


def _row_table(row: str, n: int) -> dict[str, object]:
    """Parse one of PET's row strings (``_parse_rows``) into its entry-table form."""
    parts = row.split()
    try:
        if parts[0] == "import":
            # import FILE Z_RANGE DATA [DATA2] TIME PARAM
            names = parts[3:]
            return {"taper": "import", "file": parts[1], "z_range": parts[2],
                    **_names(names)}  # fmt: skip
        # TAPER X Y Z RADIUS Z_RANGE ANISO ROTATION DATA [DATA2] TIME PARAM
        return {
            "taper": parts[0], "x": int(float(parts[1])), "y": int(float(parts[2])),
            "z": int(float(parts[3])), "radius": int(parts[4]), "z_range": parts[5],
            "aniso": float(parts[6]), "rotation": float(parts[7]), **_names(parts[8:]),
        }  # fmt: skip
    except (IndexError, ValueError) as error:
        raise ConfigError(f"PET distance_loc row {n} cannot be read: {row!r}") from error


def _names(parts: list[str]) -> dict[str, object]:
    if len(parts) == 3:
        return {"data_type": parts[0], "time": parts[1], "param": parts[2]}
    if len(parts) == 4:  # a two-word data type, e.g. "WOPR PRO1"
        return {"data_type": f"{parts[0]} {parts[1]}", "time": parts[2], "param": parts[3]}
    raise IndexError(parts)


def _z_range(value: object, n: int) -> str | int:
    if str(value).strip() == ":":
        return "all"
    try:
        return int(str(value))
    except ValueError as error:
        raise ConfigError(f"PET distance_loc entry {n}: z_range {value!r}") from error


class _Names:
    """Resolve PET's names in an entry to the case's.

    PET compares data types and variables case-insensitively, and times as numbers or as
    dates (here: their ``YYYY-MM-DD`` part).
    """

    def __init__(
        self,
        datatypes: Sequence[str],
        times: Sequence[str],
        variables: Sequence[str],
        time_labels: Mapping[str, str] | None = None,
    ) -> None:
        self.datatypes = {d.lower(): d for d in datatypes}
        self.times: dict[object, str] = {_time(t): t for t in times}
        # PET's own labels (days) name the case's converted times too.
        self.times.update({_time(label): t for label, t in (time_labels or {}).items()})
        self.variables = {v.lower(): v for v in variables}

    def __call__(self, entry: dict[str, Any]) -> dict[str, Any] | None:
        """Return ``entry`` with the case's names, or ``None`` if it names what is missing."""
        resolved = dict(entry)
        lookups = (
            ("data", self.datatypes.get(str(entry["data"]).lower())),
            ("time", self.times.get(_time(str(entry["time"])))),
            ("parameter", self.variables.get(str(entry["parameter"]).lower())),
        )
        for key, found in lookups:
            if entry[key] == "*":
                continue
            if found is None:
                return None
            resolved[key] = found
        return resolved


def _time(label: str) -> object:
    # As PET's _parse_time: a number if it is one, else a date.
    try:
        number = float(label)
    except ValueError:
        return label.strip()[:10]
    return number if math.isfinite(number) else label
