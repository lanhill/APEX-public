"""On-disk layout of a history-matching run, with atomic checkpoints for exact resume.

::

    <run>/
      config.json           resolved configuration the run started with (used by resume)
      state.json            loop position, the algorithm's scalar state and RNG state
                            (for resume); with cross validation also the stopping rule's
                            state and whether it stopped the run
      history.json          one IterationRecord per forecast
      summary.json          prior and final data mismatch and performance metrics; with cross
                            validation the selected (posterior) iteration
      events.jsonl          structured log
      run.lock              held (flock) while a process runs the run: names the holder
      checkpoints/
        perturbed_observations.npz   perturbed observations drawn once, by member: what
                              iterative methods use throughout; for fixed schedules
                              the un-inflated draw data mismatches are scored against
        iter_<k>.npz          ensemble (state), forecast (predicted), member_ids of every
                              step (fixed schedules) or accepted iterate (iterative
                              methods); for fixed schedules, steps k >= 1 also hold the
                              perturbed_observations their analysis drew (ES-MDA:
                              re-perturbed with the inflated C_d each step); for GIES
                              (RLM-MAC) the mean-model forecast (reference_prediction);
                              raw_predicted too with [output] save_raw_forecasts
                              with [validation] predicted holds the
                              history-matching data only, cv_predicted and
                              test_predicted the held-out data
        trial_<k>.npz         rejected trials, with [output] save_rejected_trials
        data_mismatch_<k>.npz every forecast's per-member data mismatch, its share per data type,
                              and performance metrics as metric:<key>:<label>
        algorithm_state_<n>.npz   the algorithm's loop state after checkpoint n,
                              for resume, not a result: e.g. the members still in the
                              run, the mean-model forecast, a localization taper drawn
                              from the prior; named in state.json, older ones deleted
        diagnostics_<k>.npz   optional internals of the update producing iteration k
                              (gain, taper, singular values, ...), saved only when
                              requested by [output] diagnostics or a plugin's own
                              diagnostics table (which overrides [output]); "index" says
                              what was saved or skipped (over max_mb)
        diagnostics_plan.json what the run saves, and which setting decided each item
        data_split.json       with [validation]: the keys and counts of each role
                              (history matching, cv, test, unused) and the stopping rule
        truth.npz             the true model of a synthetic case (the run's own copy)
        posterior.npz         final ensemble per variable and its forecast
      simulations/iter_<k>/member_<id>/   simulator work directories (if the simulator
                                          needs files); a failed member's holds
                                          APEX_FAILED (the reason); which are kept is
                                          [output] keep_simulations (all by default)
      agents/               traces of runtime agents about the run: each
                            model call, decision and action as JSON, e.g. the run monitor's
"""

import dataclasses
import json
import os
import shutil
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from apex.core.errors import WorkflowError
from apex.domain.results import IterationRecord

#: Version of new runs' configuration snapshots (see :meth:`RunDirectory.snapshot_version`).
SNAPSHOT_VERSION = 5

#: Metrics renamed on 2026-09-30: the mean-square
#: data mismatch became ``normalized_data_mismatch``, and ``data_mismatch`` now names what
#: was the data misfit, the objective the methods minimise.
RENAMED_METRICS = {
    "data_mismatch": "normalized_data_mismatch",
    "cv_data_mismatch": "cv_normalized_data_mismatch",
    "test_data_mismatch": "test_normalized_data_mismatch",
}
#: Fields of old history records and score files, by their new names.
RENAMED_FIELDS = {
    "misfit_mean": "data_mismatch_mean",
    "misfit_std": "data_mismatch_std",
    "misfit_by_type": "data_mismatch_by_type",
}


def upgrade_metric_name(name: str) -> str:
    """Return a metric's current name for the name an old run (before the rename) used."""
    return RENAMED_METRICS.get(name, name)


def upgrade_record(record: dict[str, Any]) -> dict[str, Any]:
    """Return a history record in the current names (an old one has ``misfit_mean``).

    Old records name the objective ``misfit`` and the mean-square metric ``data_mismatch``;
    detected per record, so a resumed old run (old and new records) reads consistently.
    """
    if "misfit_mean" not in record:
        return record
    upgraded = {RENAMED_FIELDS.get(k, k): v for k, v in record.items()}
    metrics = {upgrade_metric_name(k): v for k, v in (upgraded.get("metrics") or {}).items()}
    # New runs record the objective as a metric too, from the same means: add it, so that
    # studies and summaries find it for old runs as well (bit-identical to a new record).
    metrics["data_mismatch"] = {
        "all": upgraded["data_mismatch_mean"],
        **(upgraded.get("data_mismatch_by_type") or {}),
    }
    upgraded["metrics"] = metrics
    return upgraded


def upgrade_scores(arrays: dict[str, Any]) -> dict[str, Any]:
    """Return an old ``misfit_<k>.npz``'s arrays in the current names."""
    out: dict[str, Any] = {}
    # The objective as a metric, as new runs save it (overall and per data type).
    if "misfit" in arrays:
        out["metric:data_mismatch:all"] = arrays["misfit"]
    if "misfit_by_type" in arrays and "datatypes" in arrays:
        for row, name in enumerate(arrays["datatypes"]):
            out[f"metric:data_mismatch:{name}"] = arrays["misfit_by_type"][row]
    for key, value in arrays.items():
        if key == "misfit":
            key = "data_mismatch"
        elif key == "misfit_by_type":
            key = "data_mismatch_by_type"
        elif key.startswith("metric:"):
            _, metric, label = key.split(":", 2)
            key = f"metric:{upgrade_metric_name(metric)}:{label}"
        out[key] = value
    return out


class RunDirectory:
    """Files of one run. Every write is atomic (temporary file, then rename).

    Parameters
    ----------
    path
        The run directory; created by :meth:`initialise`.
    """

    def __init__(self, path: Path) -> None:
        self.path = Path(path)

    @property
    def checkpoints(self) -> Path:
        """Directory of ``.npz`` checkpoints."""
        return self.path / "checkpoints"

    def initialise(self, config_text: str) -> None:
        """Create a new run directory holding the resolved configuration (JSON).

        Raises
        ------
        WorkflowError
            If the directory already contains a run (use resume instead).
        """
        if (self.path / "state.json").exists():
            raise WorkflowError(
                f"{self.path} already holds a run: resume it (apex resume {self.path}), or "
                "start a new one in another folder (another [run] name, or [run] output_dir, "
                "in the configuration), or after moving this folder away"
            )
        self.checkpoints.mkdir(parents=True, exist_ok=True)
        self._write_text(self.path / "config.json", config_text)
        self._write_text(self.path / "snapshot_version", f"{SNAPSHOT_VERSION}\n")
        self._write_text(self.path / "history.json", "[]\n")

    def snapshot_version(self) -> int:
        """Version of the run's configuration snapshot (1 for runs without the marker).

        Version 2 (2026-09-28): ``"adaptive"`` localization is the prior-only method.
        Version 3 (2026-09-30): ``data_mismatch`` is the objective.
        Version 4 (2026-10-13): rlmmac's damping ``reduction`` defaults to 0.9; older
        snapshots without one divided by the factor.
        Version 5 (2026-10-19): the ``lm`` scheme's ``confirm_convergence`` defaults to
        true; older snapshots stopped on the first small improvement (PET's rule).
        """
        marker = self.path / "snapshot_version"
        return int(marker.read_text().strip()) if marker.exists() else 1

    def config_text(self) -> str:
        """Return the configuration snapshot the run started with."""
        return (self.path / "config.json").read_text()

    #: Written into a member's work directory when its run fails (kept by retention).
    FAILED_MARKER = "APEX_FAILED"

    def mark_failed(self, iteration: int, member_id: int, message: str) -> None:
        """Mark a member's work directory as failed (if it exists), with the reason."""
        directory = self.simulation_dir(iteration, member_id)
        if directory.is_dir():
            (directory / self.FAILED_MARKER).write_text(message + "\n")

    def prune_simulations(self, current: int, policy: str) -> None:
        """Apply a retention policy to ``simulations/`` after a forecast of ``current``.

        ``failed``/``none`` prune the current forecast's directories (keeping failed
        members' for ``failed``); ``last``/``failed_and_last`` prune earlier forecasts'
        (keeping failed members' for ``failed_and_last``). ``all`` keeps everything.

        Raises
        ------
        ValueError
            For an unknown policy.
        """
        pruned: set[Path] = set()
        for member in self.simulations_to_prune(current, policy):
            if member.is_dir():
                shutil.rmtree(member)
            else:
                member.unlink()
            pruned.add(member.parent)
        for iteration_dir in pruned:  # only the forecasts pruned now, emptied
            if not any(iteration_dir.iterdir()):
                iteration_dir.rmdir()

    def simulations_to_prune(self, current: int, policy: str) -> list[Path]:
        """Return what :meth:`prune_simulations` would remove (see it), sorted.

        For an executor whose members' files are elsewhere: it removes them
        there first, from the markers in these folders.

        Raises
        ------
        ValueError
            For an unknown policy.
        """
        if policy not in ("all", "failed", "last", "failed_and_last", "none"):
            raise ValueError(f"unknown retention policy {policy!r}")
        root = self.path / "simulations"
        if policy == "all" or not root.is_dir():
            return []
        found: list[Path] = []
        for iteration_dir in sorted(root.glob("iter_*")):
            is_current = iteration_dir.name == f"iter_{current}"
            if is_current and policy in ("last", "failed_and_last"):
                continue
            if not is_current and policy in ("failed", "none"):
                continue  # pruned when it was current
            keep_failed = policy in ("failed", "failed_and_last")
            for member in sorted(iteration_dir.iterdir()):
                if keep_failed and (member / self.FAILED_MARKER).exists():
                    continue
                found.append(member)
        return found

    def simulation_dir(self, iteration: int, member_id: int) -> Path:
        """Work directory for one forward run (``member_id = -1`` is the mean model)."""
        member = "mean" if member_id < 0 else f"member_{member_id}"
        return self.path / "simulations" / f"iter_{iteration}" / member

    def save_arrays(self, name: str, arrays: Mapping[str, NDArray[Any]]) -> None:
        """Write ``checkpoints/<name>.npz`` atomically."""
        self.checkpoints.mkdir(parents=True, exist_ok=True)
        target = self.checkpoints / f"{name}.npz"
        with tempfile.NamedTemporaryFile(dir=self.checkpoints, suffix=".npz", delete=False) as tmp:
            # Typed as Any: numpy's stubs confuse **arrays with the allow_pickle keyword.
            payload: dict[str, Any] = dict(arrays)
            np.savez(tmp, **payload)
        os.replace(tmp.name, target)

    def load_arrays(self, name: str) -> dict[str, NDArray[Any]]:
        """Read ``checkpoints/<name>.npz``.

        ``data_mismatch_<k>`` falls back to an old run's ``misfit_<k>.npz``, read in the
        current names (:func:`upgrade_scores`).
        """
        path = self.checkpoints / f"{name}.npz"
        if not path.exists() and name.startswith("data_mismatch_"):
            old = self.checkpoints / f"misfit_{name.removeprefix('data_mismatch_')}.npz"
            if old.exists():
                with np.load(old) as data:
                    return upgrade_scores({key: data[key] for key in data.files})
        with np.load(path) as data:
            return {key: data[key] for key in data.files}

    def has_arrays(self, name: str) -> bool:
        """Whether ``checkpoints/<name>.npz`` exists (or an old run's ``misfit_<k>.npz``)."""
        if (self.checkpoints / f"{name}.npz").exists():
            return True
        rest = name.removeprefix("data_mismatch_")
        return rest != name and (self.checkpoints / f"misfit_{rest}.npz").exists()

    def write_state(self, state: Mapping[str, Any]) -> None:
        """Replace ``state.json``; written last in each step, so it marks what is complete."""
        self._write_text(self.path / "state.json", json.dumps(state, indent=2) + "\n")

    def read_state(self) -> dict[str, Any]:
        """Read ``state.json``.

        Raises
        ------
        WorkflowError
            If the run has no saved state.
        """
        path = self.path / "state.json"
        if not path.exists():
            raise WorkflowError(f"{self.path} has no saved state to resume from")
        state: dict[str, Any] = json.loads(path.read_text())
        return state

    def append_history(self, record: IterationRecord) -> None:
        """Add one record to ``history.json``."""
        history = self.history_dicts()
        history.append(dataclasses.asdict(record))
        self._write_text(self.path / "history.json", json.dumps(history, indent=2) + "\n")

    def history(self) -> list[IterationRecord]:
        """Return all records written so far."""
        return [IterationRecord(**entry) for entry in self.history_dicts()]

    def history_dicts(self) -> list[dict[str, Any]]:
        """``history.json`` as plain dictionaries."""
        path = self.path / "history.json"
        raw: list[dict[str, Any]] = json.loads(path.read_text()) if path.exists() else []
        records = [upgrade_record(r) for r in raw]
        return records

    @property
    def stop_file(self) -> Path:
        """A request to stop at the next checkpoint; removed on resume."""
        return self.path / "stop_requested"

    def request_stop(self, by: str | None = None) -> None:
        """Ask the running workflow to stop at its next checkpoint (``by``: who asks)."""
        self.path.mkdir(parents=True, exist_ok=True)
        asker = f" ({by})" if by else ""
        self.stop_file.write_text(f"stop at the next checkpoint{asker}\n")

    def stop_requested_by(self, by: str) -> bool:
        """Whether the stop request is the one ``by`` made (:meth:`request_stop`)."""
        try:
            return f"({by})" in self.stop_file.read_text()
        except OSError:
            return False

    def stop_requested(self) -> bool:
        """Whether a stop has been requested."""
        return self.stop_file.exists()

    def clear_stop(self) -> None:
        """Withdraw a stop request (before resuming)."""
        self.stop_file.unlink(missing_ok=True)

    def truncate_history(self, n_records: int) -> None:
        """Keep the first ``n_records`` records (drops entries written after a checkpoint)."""
        history = self.history_dicts()[:n_records]
        self._write_text(self.path / "history.json", json.dumps(history, indent=2) + "\n")

    @staticmethod
    def _write_text(target: Path, text: str) -> None:
        target.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            "w", dir=target.parent, suffix=".tmp", delete=False, encoding="utf-8"
        ) as tmp:
            tmp.write(text)
        os.replace(tmp.name, target)
