"""On-disk layout of a surrogate build and of its checks.

::

    runs/<name>/
      config.json          the resolved SurrogateConfig
      state.json           the build's fingerprint, its streams' states, complete
      provenance.json      APEX version and commit, libraries, host, input hashes
      design.npz           the design: plan, member ids and inputs (state space) of the
                           held-out and the training runs (and, for plan "run", their outputs)
      members/member_<id>.npz   one finished full run: values (n_keys,), seconds
      snapshots.jsonl      one line per finished member: ok, or failed with its message
      snapshots.npz        the assembled SnapshotSet (the surrogate's inputs, the outputs)
      simulations/member_<id>/   the simulator's work directories ([output] keep_simulations)
      model/surrogate.json, model/surrogate.npz   the fitted surrogate (storage/surrogate_files)
      gate.json            the gate's verdict on the held-out runs
      summary.json         what the build did (the summarizer reads it)
      events.jsonl         structured log
      run.lock             one process per directory
      checks/<id>/         apex surrogate check: check.json, gate.json, design.npz,
                           members/, snapshots.jsonl, simulations/, run.lock

Every file is written to a temporary name and renamed, so an interrupted build never leaves
a partial member that a resume would take as finished. A resume runs only the members
without a file in ``members/`` and not recorded as failed in ``snapshots.jsonl``. A
successful member is saved by the worker that ran it as soon as it finishes; failures are
journalled by the main process when the batch's ``run_all`` returns, so a member that failed
in a batch interrupted before then is run again on resume.
"""

import json
import os
import shutil
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from apex.core.errors import ConfigError
from apex.domain.observations import DataKey
from apex.domain.state import StateLayout, StateVariable
from apex.domain.surrogate import SnapshotProvenance, SnapshotSet, key_document

MODEL = Path("model") / "surrogate.json"
SNAPSHOTS = "snapshots.npz"
JOURNAL = "snapshots.jsonl"
DESIGN = "design.npz"
GATE = "gate.json"
SUMMARY = "summary.json"
CHECK = "check.json"
FAILED_MARKER = "APEX_FAILED"

#: What ``state.json`` names a build or a check by.
BUILD_FINGERPRINT = "surrogate_fingerprint"
CHECK_FINGERPRINT = "surrogate_check_fingerprint"


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def _save_npz(path: Path, arrays: Mapping[str, NDArray[Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.npz")
    payload: dict[str, Any] = dict(arrays)
    np.savez(temporary, **payload)
    os.replace(temporary, path)


@dataclass(frozen=True)
class MemberRecord:
    """One finished member of a design, from the journal and ``members/``."""

    member_id: int
    ok: bool
    seconds: float | None = None
    message: str | None = None


@dataclass(frozen=True)
class StoredDesign:
    """A design as ``design.npz`` holds it.

    Attributes
    ----------
    plan
        ``prior``, ``lhs``, ``file`` or ``run`` (a check's: ``prior`` or ``posterior``).
    held_out_ids, training_ids
        The members of each set.
    held_out, training
        ``{variable: (size, n)}`` in state space.
    held_out_outputs, training_outputs
        For plan ``run``: the stored forecasts ``(n_keys, n)``; else ``None``.
    failed_ids
        For plan ``run``: the members whose stored forecasts are not finite (left out).
    """

    plan: str
    held_out_ids: tuple[int, ...]
    training_ids: tuple[int, ...]
    held_out: Mapping[str, NDArray[np.float64]]
    training: Mapping[str, NDArray[np.float64]]
    held_out_outputs: NDArray[np.float64] | None = None
    training_outputs: NDArray[np.float64] | None = None
    failed_ids: tuple[int, ...] = ()


class SurrogateDirectory:
    """A surrogate build's (or check's) directory (module docstring)."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)

    @property
    def members(self) -> Path:
        """Where finished members are saved."""
        return self.path / "members"

    @property
    def model(self) -> Path:
        """The fitted surrogate's manifest."""
        return self.path / MODEL

    # ---------------------------------------------------------------- state

    def read_state(self) -> dict[str, Any]:
        """Return ``state.json`` (empty if there is none)."""
        path = self.path / "state.json"
        if not path.exists():
            return {}
        state: dict[str, Any] = json.loads(path.read_text())
        return state

    def write_state(self, **fields: Any) -> None:  # noqa: ANN401 (JSON values)
        """Merge ``fields`` into ``state.json``."""
        state = self.read_state()
        state.update(fields)
        _write_text(self.path / "state.json", json.dumps(state, indent=2, sort_keys=True))

    @staticmethod
    def holds_build(path: Path) -> bool:
        """Whether ``path`` holds a surrogate build."""
        state = Path(path) / "state.json"
        return state.is_file() and BUILD_FINGERPRINT in json.loads(state.read_text())

    def open(
        self,
        *,
        kind: str,
        config_text: str,
        fingerprint: str,
        parts: Mapping[str, str] | None = None,
    ) -> bool:
        """Create the directory, or check that an existing one is this build (or check).

        ``kind`` is :data:`BUILD_FINGERPRINT` or :data:`CHECK_FINGERPRINT`; ``parts`` what
        the fingerprint is made of, saved so that a refusal names what differs. Returns
        whether it existed (resumed).

        Raises
        ------
        ConfigError
            If it holds another build (or check), or anything else.
        """
        state = self.read_state()
        if state:
            if state.get(kind) != fingerprint:
                saved = state.get("fingerprint_parts") or {}
                differ = sorted(
                    name for name in set(saved) | set(parts or {})
                    if saved.get(name) != (parts or {}).get(name)
                )  # fmt: skip
                what = f" ({', '.join(differ)} differ)" if saved and parts and differ else ""
                raise ConfigError(
                    f"{self.path} holds a surrogate build or check of other runs{what}; "
                    "choose another name or output_dir, or delete it to start again: "
                    f"rm -r '{self.path}'"
                )
            return True
        if self.path.is_dir() and any(e.name != "run.lock" for e in self.path.iterdir()):
            raise ConfigError(f"{self.path} is not empty and holds no surrogate build")
        self.members.mkdir(parents=True, exist_ok=True)
        _write_text(self.path / "config.json", config_text)
        self.write_state(**{kind: fingerprint}, fingerprint_parts=dict(parts or {}))
        return False

    def write_config(self, text: str) -> None:
        """Replace ``config.json`` atomically."""
        _write_text(self.path / "config.json", text)

    def write_json(self, name: str, content: Mapping[str, Any]) -> Path:
        """Write ``<name>`` (a ``.json`` file name) in the directory."""
        path = self.path / name
        _write_text(path, json.dumps(content, indent=2, default=str) + "\n")
        return path

    def read_json(self, name: str) -> dict[str, Any] | None:
        """Return ``<name>``'s content, or ``None`` if it is missing."""
        path = self.path / name
        if not path.is_file():
            return None
        content: dict[str, Any] = json.loads(path.read_text())
        return content

    # ---------------------------------------------------------------- design

    def save_design(self, design: StoredDesign, layout: StateLayout) -> None:
        """Write ``design.npz`` (variables by their position in ``layout``)."""
        arrays: dict[str, NDArray[Any]] = {
            "plan": np.array(design.plan),
            "variables": np.array(layout.names),
            "held_out_ids": np.asarray(design.held_out_ids, dtype=np.int64),
            "training_ids": np.asarray(design.training_ids, dtype=np.int64),
            "failed_ids": np.asarray(design.failed_ids, dtype=np.int64),
        }
        for i, name in enumerate(layout.names):
            arrays[f"held_out_{i}"] = np.ascontiguousarray(design.held_out[name])
            arrays[f"training_{i}"] = np.ascontiguousarray(design.training[name])
        if design.held_out_outputs is not None and design.training_outputs is not None:
            arrays["held_out_outputs"] = design.held_out_outputs
            arrays["training_outputs"] = design.training_outputs
        _save_npz(self.path / DESIGN, arrays)

    def load_design(self) -> StoredDesign | None:
        """Return the saved design, or ``None`` if it has not been drawn."""
        path = self.path / DESIGN
        if not path.is_file():
            return None
        with np.load(path, allow_pickle=False) as data:
            names = [str(n) for n in data["variables"]]
            outputs = "held_out_outputs" in data.files
            return StoredDesign(
                plan=str(data["plan"]),
                held_out_ids=tuple(int(i) for i in data["held_out_ids"]),
                training_ids=tuple(int(i) for i in data["training_ids"]),
                held_out={
                    n: np.ascontiguousarray(data[f"held_out_{i}"]) for i, n in enumerate(names)
                },
                training={
                    n: np.ascontiguousarray(data[f"training_{i}"]) for i, n in enumerate(names)
                },
                held_out_outputs=np.ascontiguousarray(data["held_out_outputs"])
                if outputs
                else None,
                training_outputs=np.ascontiguousarray(data["training_outputs"])
                if outputs
                else None,
                failed_ids=tuple(int(i) for i in data["failed_ids"]),
            )

    # ---------------------------------------------------------------- members

    def member_file(self, member_id: int) -> Path:
        """File of one finished member."""
        return self.members / f"member_{member_id}.npz"

    def simulation_dir(self, member_id: int) -> Path:
        """Work directory of one member's run."""
        return self.path / "simulations" / f"member_{member_id}"

    def save_member(self, member_id: int, values: NDArray[np.float64], seconds: float) -> None:
        """Save one member's outputs ``(n_keys,)`` atomically."""
        _save_npz(
            self.member_file(member_id),
            {"values": np.asarray(values, dtype=np.float64), "seconds": np.array(seconds)},
        )

    def load_member(self, member_id: int) -> tuple[NDArray[np.float64], float] | None:
        """Return one member's outputs and run time, or ``None`` if it has not finished."""
        path = self.member_file(member_id)
        if not path.is_file():
            return None
        with np.load(path, allow_pickle=False) as data:
            return np.ascontiguousarray(data["values"], dtype=np.float64), float(data["seconds"])

    def record(self, records: Sequence[MemberRecord]) -> None:
        """Append members' outcomes to ``snapshots.jsonl`` (one line each, flushed).

        A last line torn by an interruption is ended first, so the new lines stay whole
        (the torn one is skipped by :meth:`journal`).
        """
        if not records:
            return
        lines = "".join(
            json.dumps({"member": r.member_id, "ok": r.ok, "seconds": r.seconds,
                        "message": r.message}) + "\n"
            for r in records
        )  # fmt: skip
        path = self.path / JOURNAL
        if path.is_file() and path.stat().st_size:
            with path.open("rb") as existing:
                existing.seek(-1, os.SEEK_END)
                if existing.read(1) != b"\n":
                    lines = "\n" + lines
        with path.open("a", encoding="utf-8") as handle:
            handle.write(lines)
            handle.flush()
            os.fsync(handle.fileno())

    def journal(self) -> dict[int, MemberRecord]:
        """Return the journal's last record per member (a torn last line is left out)."""
        path = self.path / JOURNAL
        found: dict[int, MemberRecord] = {}
        if not path.is_file():
            return found
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue  # interrupted while writing: that member is run again
            member = int(row["member"])
            found[member] = MemberRecord(member, bool(row["ok"]), row.get("seconds"),
                                         row.get("message"))  # fmt: skip
        return found

    def mark_failed(self, member_id: int, message: str) -> None:
        """Mark a failed member's work directory, if it has one."""
        folder = self.simulation_dir(member_id)
        if folder.is_dir():
            (folder / FAILED_MARKER).write_text(message + "\n", encoding="utf-8")

    def prune_simulations(self, keep: str) -> None:
        """Remove work directories: every one (``none``), or those not failed (``failed``)."""
        root = self.path / "simulations"
        if keep == "all" or not root.is_dir():
            return
        for folder in root.iterdir():
            if keep == "none" or not (folder / FAILED_MARKER).exists():
                shutil.rmtree(folder, ignore_errors=True)

    # ---------------------------------------------------------------- snapshots

    def save_snapshots(self, snapshots: SnapshotSet) -> Path:
        """Write ``snapshots.npz``: everything :func:`load_snapshots` needs."""
        layout = [
            {"name": v.name, "size": v.size, "lower": v.lower, "upper": v.upper,
             "grid": None if v.grid is None else list(v.grid)}
            for v in snapshots.layout.variables
        ]  # fmt: skip
        arrays: dict[str, NDArray[Any]] = {
            "layout": np.array(json.dumps(layout)),
            "keys": np.array(json.dumps([key_document(k) for k in snapshots.keys])),
            "outputs": snapshots.outputs,
            "run_ids": np.asarray(snapshots.run_ids, dtype=np.int64),
            "failed": np.asarray(snapshots.failed, dtype=np.int64),
            "provenance": np.array(snapshots.provenance.model_dump_json()),
            "digest": np.array(snapshots.digest()),
        }
        for i, variable in enumerate(snapshots.layout.variables):
            arrays[f"input_{i}"] = snapshots.inputs[variable.name]
        if snapshots.coarse_outputs is not None:
            arrays["coarse_outputs"] = snapshots.coarse_outputs
        if snapshots.fields:
            # State fields, in the same file: one digest covers them.
            arrays["field_names"] = np.array(json.dumps(sorted(snapshots.fields)))
            arrays["field_times"] = np.array(json.dumps(list(snapshots.field_times)))
            for i, name in enumerate(sorted(snapshots.fields)):
                arrays[f"field_{i}"] = snapshots.fields[name]
        path = self.path / SNAPSHOTS
        _save_npz(path, arrays)
        return path


def load_snapshots(path: Path) -> SnapshotSet:
    """Read a ``snapshots.npz`` written by :meth:`SurrogateDirectory.save_snapshots`.

    Raises
    ------
    ConfigError
        If the file is not one, or its content does not match its recorded digest.
    """
    try:
        with np.load(path, allow_pickle=False) as data:
            layout = StateLayout([
                StateVariable(str(v["name"]), int(v["size"]), float(v["lower"]),
                              float(v["upper"]), None if v["grid"] is None else tuple(v["grid"]))
                for v in json.loads(str(data["layout"]))
            ])  # fmt: skip
            keys = tuple(DataKey(t, d, i) for t, d, i in json.loads(str(data["keys"])))
            names = json.loads(str(data["field_names"])) if "field_names" in data.files else []
            snapshots = SnapshotSet(
                layout=layout,
                inputs={v.name: data[f"input_{i}"] for i, v in enumerate(layout.variables)},
                keys=keys,
                outputs=data["outputs"],
                run_ids=tuple(int(i) for i in data["run_ids"]),
                failed=tuple(int(i) for i in data["failed"]),
                provenance=SnapshotProvenance.model_validate_json(str(data["provenance"])),
                coarse_outputs=data["coarse_outputs"] if "coarse_outputs" in data.files else None,
                fields={str(n): data[f"field_{i}"] for i, n in enumerate(names)},
                field_times=tuple(json.loads(str(data["field_times"]))) if names else (),
            )
            recorded = str(data["digest"])
    except (OSError, KeyError, ValueError) as error:
        raise ConfigError(f"cannot read the snapshots {path}: {error}") from None
    if snapshots.digest() != recorded:
        raise ConfigError(f"the snapshots {path} do not match their recorded digest")
    return snapshots
