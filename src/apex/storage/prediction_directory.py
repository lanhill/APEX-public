"""On-disk layout of a prediction.

::

    RUN_DIR/predictions/<name>/
      config.json          resolved PredictionConfig
      provenance.json      APEX version and commit, library versions, host, input hashes
      state.json           fingerprint of the configuration, resolved axis (vectors, times)
      members/<ensemble>_<member>.npz   one finished member (values (n_vectors, n_times))
      forecast_<ensemble>.npz  assembled: values (n_vectors, n_times, n), times, vectors,
                           member_ids, failed_member_ids, [mean_model]
      truth.npz            the true model's forecast (synthetic cases)
      statistics.csv       per ensemble, vector and time: n, mean, std, percentiles
      scores.json, scores.csv   forecast scores against the truth
      events.jsonl         structured log
      simulations/<ensemble>/<member>/   work directories

A prediction has no randomness and no iteration: resuming it forecasts only the members
without a file in ``members/``, then assembles again.
"""

import csv
import json
import logging
import os
import shutil
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from apex.core.errors import ConfigError
from apex.domain.prediction import EnsembleForecast
from apex.storage.locks import LOCK_FILE

logger = logging.getLogger(__name__)

#: Member ids of the extra forecasts of an ensemble (never ids of real members).
MEAN_MODEL_ID = -1
TRUTH_ID = -2


def ensemble_slug(name: str) -> str:
    """File-name form of an ensemble name (``iteration:3`` becomes ``iteration_3``)."""
    return name.replace(":", "_")


def member_tag(member_id: int) -> str:
    """File-name form of a member (``member_4``, ``mean`` or ``truth``)."""
    return {MEAN_MODEL_ID: "mean", TRUTH_ID: "truth"}.get(member_id, f"member_{member_id}")


class PredictionDirectory:
    """A prediction's directory (see the module docstring)."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)

    @property
    def members(self) -> Path:
        """Where finished members are saved."""
        return self.path / "members"

    @staticmethod
    def holds_prediction(path: Path) -> bool:
        """Whether ``path`` is a prediction directory (it has a ``state.json`` of one)."""
        state = Path(path) / "state.json"
        if not state.exists():
            return False
        return "prediction_fingerprint" in json.loads(state.read_text())

    def open(self, *, config_text: str, fingerprint: str) -> bool:
        """Create the directory, or check that an existing one is this prediction.

        Returns whether it existed (the prediction is resumed). A prediction with another
        configuration is replaced when it holds no finished member and nothing else.

        Raises
        ------
        ConfigError
            If it holds a prediction with another configuration that cannot be replaced.
        """
        state_path = self.path / "state.json"
        # A stop asked before (it ended a job, or was ignored) is not one for this start.
        (self.path / "stop_requested").unlink(missing_ok=True)
        if state_path.exists():
            state = json.loads(state_path.read_text())
            if state.get("prediction_fingerprint") != fingerprint:
                # Finished members are never deleted automatically, not even after a failed
                # history check (their forecasts may still be wanted): only a folder with
                # none is replaced.
                finished = self.finished_members()
                if finished or not self._clear_unfinished():
                    held = f"{len(finished)} finished member(s)" if finished else "other files"
                    if state.get("history_check") == "failed":
                        held += " (its history check failed)"
                    raise ConfigError(
                        f"{self.path} holds a prediction with another configuration and "
                        f"{held}; choose another name or output_dir, or delete it to start "
                        f"again with this configuration: rm -r '{self.path}'"
                    )
                logger.warning(
                    "replaced the prediction in %s: another configuration, no finished member",
                    self.path,
                    extra={"event": "prediction_replaced", "path": str(self.path)},
                )
            else:
                return True
        self.members.mkdir(parents=True, exist_ok=True)
        _write_text(self.path / "config.json", config_text)
        _write_text(state_path, json.dumps({"prediction_fingerprint": fingerprint}, indent=2))
        return False

    def finished_members(self) -> list[Path]:
        """Return the files of the members that finished (``members/*.npz``)."""
        return sorted(self.members.glob("*.npz")) if self.members.is_dir() else []

    def _clear_unfinished(self) -> bool:
        """Remove a prediction's files when no member finished; return whether it could.

        A prediction that failed before its first member has nothing worth keeping, so a
        corrected configuration replaces it instead of being refused (the release tester,
        2026-10-19). Only this
        layout's own entries are removed, and the lock (``run.lock``, held by this process)
        stays; a folder with anything else in it is left alone.
        """
        entries = [e for e in self.path.iterdir() if e.name != LOCK_FILE]
        if any(not _layout_entry(e.name) for e in entries):
            return False
        for entry in entries:
            if entry.is_dir() and not entry.is_symlink():
                shutil.rmtree(entry)
            else:
                entry.unlink()
        return True

    def read_state(self) -> dict[str, Any]:
        """Return ``state.json``."""
        state: dict[str, Any] = json.loads((self.path / "state.json").read_text())
        return state

    def write_state(self, **fields: Any) -> None:  # noqa: ANN401 (JSON values)
        """Merge ``fields`` into ``state.json``."""
        state = self.read_state()
        state.update(fields)
        _write_text(self.path / "state.json", json.dumps(state, indent=2))

    def write_json(self, name: str, content: Mapping[str, Any]) -> Path:
        """Write ``<name>.json``."""
        path = self.path / f"{name}.json"
        _write_text(path, json.dumps(content, indent=2, default=str) + "\n")
        return path

    def simulation_dir(self, ensemble: str, member_id: int) -> Path:
        """Work directory of one member's run."""
        return self.path / "simulations" / ensemble_slug(ensemble) / member_tag(member_id)

    def member_file(self, ensemble: str, member_id: int) -> Path:
        """File of one finished member."""
        return self.members / f"{ensemble_slug(ensemble)}_{member_tag(member_id)}.npz"

    def save_member(self, ensemble: str, member_id: int, values: NDArray[np.float64]) -> None:
        """Save one member's forecast ``(n_vectors, n_times)`` atomically."""
        _save_npz(self.member_file(ensemble, member_id), {"values": values})

    def load_member(self, ensemble: str, member_id: int) -> NDArray[np.float64] | None:
        """Load one member's forecast, or ``None`` if it has not finished."""
        path = self.member_file(ensemble, member_id)
        if not path.exists():
            return None
        with np.load(path) as data:
            values: NDArray[np.float64] = data["values"]
        return values

    def save_forecast(self, forecast: EnsembleForecast) -> None:
        """Save an assembled ensemble forecast (``forecast_<ensemble>.npz``)."""
        arrays: dict[str, NDArray[Any]] = {
            "values": forecast.values,
            "times": np.array(forecast.times),
            "vectors": np.array(forecast.vectors),
            "member_ids": forecast.member_ids,
            "failed_member_ids": forecast.failed_member_ids,
        }
        if forecast.mean_model is not None:
            arrays["mean_model"] = forecast.mean_model
        _save_npz(self.path / f"forecast_{ensemble_slug(forecast.name)}.npz", arrays)

    def load_forecast(self, name: str) -> EnsembleForecast:
        """Load an assembled ensemble forecast.

        Raises
        ------
        FileNotFoundError
            If it has not been assembled.
        """
        with np.load(self.path / f"forecast_{ensemble_slug(name)}.npz") as data:
            return EnsembleForecast(
                name=name,
                times=tuple(str(t) for t in data["times"]),
                vectors=tuple(str(v) for v in data["vectors"]),
                values=data["values"],
                member_ids=data["member_ids"],
                failed_member_ids=data["failed_member_ids"],
                mean_model=data["mean_model"] if "mean_model" in data.files else None,
            )

    def save_truth(
        self, values: NDArray[np.float64], times: Sequence[str], vectors: Sequence[str]
    ) -> None:
        """Save the true model's forecast ``(n_vectors, n_times)``."""
        _save_npz(
            self.path / "truth.npz",
            {"values": values, "times": np.array(times), "vectors": np.array(vectors)},
        )

    def load_truth(self) -> NDArray[np.float64] | None:
        """Load the true model's forecast, or ``None``."""
        path = self.path / "truth.npz"
        if not path.exists():
            return None
        with np.load(path) as data:
            values: NDArray[np.float64] = data["values"]
        return values

    def write_csv(self, name: str, rows: Sequence[Mapping[str, object]]) -> Path:
        """Write ``<name>.csv`` from dictionaries with the same keys."""
        path = self.path / f"{name}.csv"
        with path.open("w", newline="", encoding="utf-8") as handle:
            if rows:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
        return path


#: The entries a prediction writes (the module docstring's layout), and their temporaries.
_LAYOUT_FILES = frozenset({
    "config.json", "provenance.json", "state.json", "truth.npz", "statistics.csv",
    "scores.json", "scores.csv", "events.jsonl", "stop_requested", "members", "simulations",
})  # fmt: skip


def _layout_entry(name: str) -> bool:
    base = name.removesuffix(".tmp")
    return base in _LAYOUT_FILES or (base.startswith("forecast_") and base.endswith(".npz"))


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def _save_npz(path: Path, arrays: Mapping[str, NDArray[Any]]) -> None:
    # Written to a temporary file and renamed, so that an interrupted run never leaves a
    # partial member file that a resume would take as finished.
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp.npz")
    # Typed as Any: numpy's stubs confuse **arrays with the allow_pickle keyword.
    payload: dict[str, Any] = dict(arrays)
    np.savez(temporary, **payload)
    os.replace(temporary, path)
