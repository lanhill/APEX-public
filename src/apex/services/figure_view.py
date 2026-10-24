"""A run's results as the figure plugins read them (:class:`apex.core.figures.RunView`).

The plugins live in ``reporting``, which may not read storage. This adaptor reads the run
directory once (:class:`~apex.storage.run_results.RunResults`, the analysis observations, the
grids and active cells, the deck's ``START``) and gives each figure what it asks for.
"""

import datetime as dt
import logging
from collections.abc import Mapping
from functools import cached_property
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from apex.core.errors import ApexError, ConfigError
from apex.core.figures import (
    FieldLayout,
    FieldLayoutError,
    GridView,
    IterateView,
    ObservationsView,
)
from apex.domain.config import HistoryMatchConfig
from apex.domain.observations import DataKey
from apex.storage.priors import read_array
from apex.storage.run_directory import RunDirectory
from apex.storage.run_results import RunResults

#: The metrics whose ``actnum_file`` option gives the grids' active cells.
_ACTNUM_METRICS = ("rmse", "ssim", "model_change")


logger = logging.getLogger(__name__)


class StoredRunView:
    """The :class:`~apex.core.figures.RunView` of a run directory.

    Raises
    ------
    ConfigError
        If the directory has no configuration snapshot or no scored forecast yet.
    """

    def __init__(self, run_path: Path) -> None:
        from apex.services.composition import load_results, snapshot_config

        self._path = Path(run_path)
        self.config: HistoryMatchConfig = snapshot_config(RunDirectory(self._path))
        self.results: RunResults = load_results(self._path)
        self._iterates: dict[int, IterateView] = {}

    @property
    def path(self) -> Path:
        """The run directory."""
        return self._path

    @property
    def iterations(self) -> list[int]:
        """Every iteration with a scored forecast, in order."""
        return list(self.results.iterations)

    @property
    def accepted_iterations(self) -> list[int]:
        """The accepted iterates (the prior included), in order."""
        return list(self.results.accepted_iterations)

    @property
    def final_iteration(self) -> int:
        """The last accepted iterate."""
        return int(self.results.final_iteration)

    @property
    def posterior_iteration(self) -> int:
        """The posterior: with cross validation the minimum-CV iterate, else the last."""
        return int(self.results.posterior_iteration)

    @cached_property
    def start(self) -> dt.date | None:
        """The deck's ``START`` (deck simulators), else ``None``."""
        from apex.simulation.deck import case_folder, schedule_summary

        # The options that hold the deck: a wrapped simulator's are nested.
        options = _holding(self.config.simulator.options, "deck_template_file")
        template = None if options is None else options["deck_template_file"]
        if options is None or not isinstance(template, str) or not Path(template).is_file():
            return None
        try:
            summary = schedule_summary(Path(template), case_folder(options))
        except (OSError, ValueError):
            return None
        start = summary.get("start")
        return start if isinstance(start, dt.date) else None

    @cached_property
    def _workflow(self) -> Any:  # noqa: ANN401 (the composition's workflow)
        from apex.services.composition import build_run

        return build_run(self.config).workflow

    def observations(self) -> ObservationsView:
        """Return the observations in the analysis data space."""
        observations = self._workflow.analysis_observations
        return ObservationsView(
            tuple(observations.keys),
            np.asarray(observations.values, dtype=float),
            np.asarray(observations.variances, dtype=float),
        )

    def iterate(self, iteration: int) -> IterateView:
        """Return a saved iterate, its forecast over every role's data.

        Raises
        ------
        KeyError
            If it was not saved.
        """
        if iteration in self._iterates:
            return self._iterates[iteration]
        from apex.services.composition import all_roles

        try:
            stored = self.results.iterate(iteration)
        except FileNotFoundError as error:
            raise KeyError(str(error)) from None
        workflow = self._workflow
        predicted = all_roles(stored, workflow.role_rows(), workflow.analysis_observations.n_data)
        keys = tuple(_parse_key(k) for k in stored.auxiliary_keys)
        view = IterateView(
            iteration=iteration,
            state=stored.state,
            predicted=predicted,
            member_ids=stored.member_ids,
            auxiliary_predicted=stored.auxiliary_predicted,
            auxiliary_keys=keys,
        )
        # Two kept: the figures alternate between the prior and the posterior, and a Brugge
        # iterate is about half a gigabyte.
        self._iterates = {**dict(list(self._iterates.items())[-1:]), iteration: view}
        return view

    def metric_labels(self) -> dict[str, list[str]]:
        """Return every recorded metric and its labels."""
        return self.results.metric_labels()

    def metric(
        self, name: str, label: str = "all", *, accepted_only: bool = False
    ) -> tuple[list[int], list[NDArray[np.float64]], list[bool]]:
        """Return a metric's iterations, per-member values and acceptance."""
        iterations, values, accepted = self.results.metric(name, label, accepted_only=accepted_only)
        return list(iterations), [np.asarray(v, dtype=float) for v in values], list(accepted)

    def truth(self) -> Mapping[str, NDArray[np.float64]] | None:
        """Return the true model (synthetic cases), or ``None``."""
        return self.results.truth()

    def variables(self) -> list[str]:
        """Return the state variables, in order."""
        return [v.name for v in self.results.layout.variables]

    def grid(self, variable: str) -> GridView | None:
        """Return a variable's grid (cells in Eclipse order) and active cells, or ``None``."""
        found = next((v for v in self.results.layout.variables if v.name == variable), None)
        if found is None or found.grid is None:
            return None
        actnum = self._actnum
        if actnum is not None and actnum.size != found.size:
            actnum = None  # another grid's: not this variable's
        return GridView(tuple(found.grid), "F", actnum)

    def field_layout(self, datatype: str) -> FieldLayout | None:
        """Return where a geophysical field's values sit, or ``None``."""
        model, error = self._geophysics
        if error is not None:
            raise FieldLayoutError(f"the run's geophysical model could not be rebuilt: {error}")
        if model is None:
            return None
        found = model.positions(datatype)
        if found is None:
            return None
        kind, positions = found
        if kind == "stations":
            return FieldLayout(kind, positions)
        nx, ny, _ = model.reference.shape
        return FieldLayout(kind, positions, (int(nx), int(ny)), model.column_steps(datatype))

    @cached_property
    def _geophysics(self) -> tuple[Any, str | None]:
        """The run's GeophysicsModel (``None`` without ``[geophysics]``), or why not."""
        if self.config.geophysics is None:
            return None, None
        from apex.services.composition import build_geophysics

        try:
            return build_geophysics(self.config), None
        except (ApexError, OSError) as error:
            # Its grid moved, say: no maps, the other figures still drawn, the cause told.
            logger.warning(
                "figure_geophysics_unavailable", extra={"run": str(self._path), "error": str(error)}
            )
            return None, str(error)

    @cached_property
    def _actnum(self) -> NDArray[np.bool_] | None:
        for section in self.config.metrics or []:
            path = section.options.get("actnum_file") if section.name in _ACTNUM_METRICS else None
            if isinstance(path, str):
                try:
                    values = read_array(Path(path), section.options.get("actnum_key"))
                except (OSError, ValueError):
                    continue  # moved or unreadable: every cell shown
                active: NDArray[np.bool_] = np.asarray(values).reshape(-1) != 0
                return active
        return None


def run_view(run_path: Path) -> StoredRunView:
    """Return the figure view of a run directory.

    Raises
    ------
    ConfigError
        If the run has no configuration snapshot or no scored forecast yet.
    """
    try:
        return StoredRunView(run_path)
    except ConfigError:
        raise
    except (ApexError, OSError, ValueError) as error:
        raise ConfigError(f"{run_path}: cannot read the run's results: {error}") from error


def _holding(options: Mapping[str, Any], name: str) -> Mapping[str, Any] | None:
    """Return the options that set ``name``: these, or a wrapped simulator's nested ones."""
    if name in options:
        return options
    for value in options.values():
        if isinstance(value, Mapping):
            found = _holding(value, name)
            if found is not None:
                return found
    return None


def _parse_key(text: str) -> DataKey:
    """Parse ``str(DataKey)``: ``datatype@time``, with ``[index]`` for a field component."""
    head, _, time = text.rpartition("@")
    index = None
    if time.endswith("]") and "[" in time:
        time, _, rest = time.partition("[")
        index = int(rest[:-1])
    return DataKey(time, head, index)
