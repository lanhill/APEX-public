"""Read the saved results of a history-matching run, for figures, notebooks and reports.

A run directory (see :mod:`apex.storage.run_directory`) holds, per forecast, the scored
statistics (``data_mismatch_<k>.npz``) and, per saved step, the ensemble and its forecast
(``iter_<k>.npz``; ``trial_<k>.npz`` for kept rejected trials). :class:`RunResults` puts
them back together by iteration.
"""

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from apex.domain.results import IterationRecord
from apex.domain.state import StateLayout
from apex.storage.run_directory import RunDirectory


@dataclass(frozen=True, slots=True)
class Iterate:
    """One saved ensemble and its forecast.

    Attributes
    ----------
    iteration
        The iteration that produced it (0 = prior).
    state
        ``{variable: (size, n_members)}``.
    predicted
        Forecast in the analysis data space, ``(n_data, n_members)``.
    member_ids
        Member id of each column.
    raw_predicted
        Untransformed forecast, when the run saved it (``[output] save_raw_forecasts``).
    perturbed_observations
        For fixed schedules (ES-MDA), the perturbed observations the step's analysis drew.
    reference_prediction
        For GIES (RLM-MAC), the forecast of the ensemble-mean model, ``(n_data,)``.
    cv_predicted, test_predicted
        With cross validation, the forecast of the held-out CV and test data,
        ``(n_cv, n_members)`` and ``(n_test, n_members)``; ``predicted`` then holds the
        history-matching data only.
    auxiliary_predicted, auxiliary_keys
        Forecasts of the simulator's auxiliary keys, never assimilated, e.g. a model-error
        model's residuals, ``(n_auxiliary, n_members)``, and their keys as text
        (``"MEC:WOPR:P1@2008-01-02"``).
    """

    iteration: int
    state: Mapping[str, NDArray[np.float64]]
    predicted: NDArray[np.float64]
    member_ids: NDArray[np.int64]
    raw_predicted: NDArray[np.float64] | None = None
    perturbed_observations: NDArray[np.float64] | None = None
    reference_prediction: NDArray[np.float64] | None = None
    cv_predicted: NDArray[np.float64] | None = None
    test_predicted: NDArray[np.float64] | None = None
    auxiliary_predicted: NDArray[np.float64] | None = None
    auxiliary_keys: tuple[str, ...] = ()


class RunResults:
    """The saved results of a run, by iteration.

    Parameters
    ----------
    path
        The run directory.
    layout
        Its state layout, to split ensembles into variables.
    """

    def __init__(self, path: Path, layout: StateLayout) -> None:
        self.run_dir = RunDirectory(path)
        self.layout = layout
        self.history: list[IterationRecord] = self.run_dir.history()
        if not self.history:
            raise ValueError(f"{path} has no scored forecast yet")

    @property
    def path(self) -> Path:
        """The run directory."""
        return self.run_dir.path

    @property
    def iterations(self) -> list[int]:
        """Iterations of every scored forecast (rejected trials included), in order."""
        return [r.iteration for r in self.history]

    @property
    def accepted_iterations(self) -> list[int]:
        """Iterations whose ensemble was kept (every step of a fixed schedule)."""
        return [r.iteration for r in self.history if r.accepted]

    @property
    def selected_iteration(self) -> int | None:
        """With cross validation, the iterate of lowest CV normalized data mismatch (the posterior).

        Read from ``summary.json``; ``None`` for runs without cross validation or not
        finished.
        """
        path = self.run_dir.path / "summary.json"
        if not path.exists():
            return None
        validation = json.loads(path.read_text()).get("validation")
        return None if validation is None else int(validation["selected_iteration"])

    @property
    def posterior_iteration(self) -> int:
        """The iterate the posterior is: the CV-selected one, else the last kept one."""
        selected = self.selected_iteration
        return self.final_iteration if selected is None else selected

    @property
    def final_iteration(self) -> int:
        """The last kept iteration."""
        return self.accepted_iterations[-1]

    def iterate(self, iteration: int) -> Iterate:
        """Load the kept ensemble of ``iteration`` (``iter_<k>.npz``).

        Raises
        ------
        FileNotFoundError
            If that iteration was not kept (a rejected trial, or not reached).
        """
        return self._load(f"iter_{iteration}", iteration)

    def trial(self, iteration: int) -> Iterate:
        """Load a rejected trial's ensemble (``trial_<k>.npz``, if the run kept trials)."""
        return self._load(f"trial_{iteration}", iteration)

    def _load(self, name: str, iteration: int) -> Iterate:
        path = self.run_dir.checkpoints / f"{name}.npz"
        if not path.exists():
            raise FileNotFoundError(f"{self.path} has no saved ensemble {name}")
        arrays = self.run_dir.load_arrays(name)
        return Iterate(
            iteration=iteration,
            state=self.layout.split(arrays["state"]),
            predicted=arrays["predicted"],
            member_ids=arrays["member_ids"],
            raw_predicted=arrays.get("raw_predicted"),
            perturbed_observations=arrays.get("perturbed_observations"),
            reference_prediction=arrays.get("reference_prediction"),
            cv_predicted=arrays.get("cv_predicted"),
            test_predicted=arrays.get("test_predicted"),
            auxiliary_predicted=arrays.get("auxiliary_predicted"),
            auxiliary_keys=tuple(str(k) for k in arrays.get("auxiliary_keys", ())),
        )

    def per_member(self, iteration: int) -> dict[str, NDArray[np.float64]]:
        """Return everything scored for one forecast (``data_mismatch_<k>.npz``)."""
        return self.run_dir.load_arrays(f"data_mismatch_{iteration}")

    def diagnostics_iterations(self) -> list[int]:
        """Return the iterations with saved diagnostics (``diagnostics_<k>.npz``)."""
        found = (p.stem.removeprefix("diagnostics_") for p in self._diagnostics_files())
        return sorted(int(k) for k in found)

    def diagnostics(self, iteration: int) -> tuple[dict[str, NDArray[Any]], dict[str, str]]:
        """Diagnostics saved for the update producing ``iteration``: ``(items, index)``.

        ``items`` maps ``"role.name"`` (e.g. ``"localization.taper"``) to its array;
        ``index`` records, for every planned item offered, whether it was saved or why it
        was skipped (e.g. over ``max_mb``). ``checkpoints/diagnostics_plan.json`` holds the
        plan and which setting (plugin switch or ``[output]``) decided each item.

        Raises
        ------
        FileNotFoundError
            If no diagnostics were saved for ``iteration``.
        """
        path = self.run_dir.checkpoints / f"diagnostics_{iteration}.npz"
        if not path.exists():
            raise FileNotFoundError(
                f"{self.path} has no diagnostics for iteration {iteration}; they are saved "
                "only when requested ([output] diagnostics or a plugin's diagnostics table)"
            )
        arrays = self.run_dir.load_arrays(f"diagnostics_{iteration}")
        index: dict[str, str] = json.loads(str(arrays.pop("index")))
        return arrays, index

    def _diagnostics_files(self) -> list[Path]:
        return [
            p
            for p in self.run_dir.checkpoints.glob("diagnostics_*.npz")
            if p.stem.removeprefix("diagnostics_").isdigit()
        ]

    def metric_labels(self) -> dict[str, list[str]]:
        """``{metric: [label, ...]}`` recorded by the run, ``"data_mismatch"`` included.

        Over every forecast: a metric such as ``model_change`` starts after the prior.
        """
        labels: dict[str, list[str]] = {"data_mismatch": ["all"]}
        for record in self.history:
            for key in self.per_member(record.iteration):
                if key.startswith("metric:"):
                    _, metric, label = key.split(":", 2)
                    if label not in labels.setdefault(metric, []):
                        labels[metric].append(label)
        return labels

    def metric(
        self, metric: str, label: str = "all", *, accepted_only: bool = False
    ) -> tuple[list[int], list[NDArray[np.float64]], list[bool]]:
        """Per-member values of a metric at every forecast: ``(iterations, values, accepted)``.

        ``metric = "data_mismatch"`` gives the data mismatch the methods minimise. Leading forecasts
        for which the metric is not defined (the prior's ``model_change``) are left out.

        Raises
        ------
        KeyError
            If the run did not record that metric and label at any forecast, or stopped
            recording it after it started (an inconsistent run directory).
        """
        key = f"metric:{metric}:{label}"
        iterations, values, accepted = [], [], []
        found = False
        for record in self.history:
            arrays = self.per_member(record.iteration)
            if key not in arrays:
                if found:
                    raise KeyError(
                        f"{self.path} recorded {metric!r} for {label!r} but not at iteration "
                        f"{record.iteration}"
                    )
                continue
            found = True
            if accepted_only and not record.accepted:
                continue
            iterations.append(record.iteration)
            values.append(np.asarray(arrays[key], dtype=np.float64))
            accepted.append(record.accepted)
        if not found:
            raise KeyError(f"{self.path} recorded no {metric!r} for {label!r}")
        return iterations, values, accepted

    def metric_means(self, metric: str, label: str = "all") -> tuple[list[int], list[float]]:
        """Member means of a metric over the kept iterations: ``(iterations, means)``."""
        iterations, values, _ = self.metric(metric, label, accepted_only=True)
        return iterations, [float(v.mean()) for v in values]

    def truth(self) -> dict[str, NDArray[np.float64]] | None:
        """Load the run's copy of the true model (synthetic cases), ``{variable: (size,)}``."""
        if not (self.run_dir.checkpoints / "truth.npz").exists():
            return None
        vector = self.run_dir.load_arrays("truth")["truth"]
        return {name: block[:, 0] for name, block in self.layout.split(vector[:, None]).items()}

    def metric_summary(self) -> list[dict[str, object]]:
        """Statistics of every metric at every kept iteration (PET-fork's metric tables).

        One row per metric, label and iteration: mean, std, median, 10th and 90th
        percentiles over the members.
        """
        rows: list[dict[str, object]] = []
        for metric, labels in self.metric_labels().items():
            for label in labels:
                iterations, values, _ = self.metric(metric, label, accepted_only=True)
                for iteration, v in zip(iterations, values, strict=True):
                    rows.append(
                        {
                            "metric": metric,
                            "label": label,
                            "iteration": iteration,
                            "mean": float(v.mean()),
                            "std": float(v.std(ddof=1)) if v.size > 1 else 0.0,
                            "median": float(np.median(v)),
                            "p10": float(np.percentile(v, 10)),
                            "p90": float(np.percentile(v, 90)),
                        }
                    )
        return rows
