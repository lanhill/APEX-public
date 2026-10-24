"""ERT's iterations as an APEX run directory (``apex ert import``).

An exported case's ERT runs leave, per iteration ``k`` and realization ``i``, the fields ERT
wrote (``runs/realization-<i>/iter-<k>/<field>.grdecl``) and the simulator's summary. The
import **replays** them through APEX's own history-matching workflow:

- :class:`ErtReplay`, an :class:`~apex.algorithms.data_assimilation.algorithm.EnsembleAlgorithm`,
  requests a forecast of each ERT iterate in turn and keeps it as an iterate;
- :class:`StoredResponses`, a simulator, answers each request with the summary ERT simulated
  for exactly that realization's parameters (found by the parameters' digest).

So the run directory is what an APEX run writes, scored by APEX: data mismatch (against
APEX-drawn perturbations of the un-inflated ``C_d``, as APEX scores ES-MDA), performance
metrics, the held-out (``cv``/``test``) data's scores, with ``[validation]`` the ``select``
rule's minimum-CV posterior, and ``summary.json``. ``apex report``, ``apex summary``, the
comparisons and the agents then read ERT's runs as APEX's. A realization ERT has no summary
for (it failed) fails in the replay too, under the run's failure policy.
"""

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from apex.algorithms.data_assimilation.algorithm import (
    AlgorithmState,
    Estimate,
    ForecastRequest,
    Forecasts,
    Record,
    RunSetup,
    Status,
    Step,
    keep,
)
from apex.core.errors import ConfigError, SimulationError
from apex.core.simulation import MemberInput
from apex.domain.config import PluginSection
from apex.domain.observations import DataKey
from apex.domain.results import HistoryMatchResult
from apex.execution.local import SerialExecutor
from apex.integrations.ert.render import read_scalar_state
from apex.services import composition
from apex.simulation.deck import read_summary
from apex.simulation.grdecl import read_grdecl


def state_digest(state: Mapping[str, NDArray[np.float64]]) -> str:
    """Return a digest of a member's parameters, by variable name (float64 values)."""
    digest = hashlib.sha256()
    for name in sorted(state):
        digest.update(name.encode())
        digest.update(np.ascontiguousarray(state[name], dtype=np.float64).tobytes())
    return digest.hexdigest()


class StoredResponses:
    """A simulator returning the summary ERT simulated for a member's exact parameters.

    ``cases`` maps a parameter digest (:func:`state_digest`) to the run path's case (the
    path without extension, ``<run path>/<ECLBASE>``).
    """

    def __init__(self, cases: Mapping[str, Path]) -> None:
        self.cases = dict(cases)

    def run(self, member: MemberInput, keys: Sequence[DataKey]) -> NDArray[np.float64]:
        """Return the stored values of ``keys``.

        Raises
        ------
        SimulationError
            If ERT has no run for these parameters, or its summary lacks a key (ERT's
            realization failed).
        """
        case = self.cases.get(state_digest(member.state))
        if case is None:
            raise SimulationError(f"member {member.member_id}: ERT has no run of these parameters")
        return read_summary(case, keys)


@dataclass(frozen=True)
class ErtIterate:
    """One ERT iteration: the state ``(n_state, n)`` of its realizations, in order."""

    iteration: int
    state: NDArray[np.float64]
    realizations: NDArray[np.int64]


class ErtReplay:
    """Keeps ERT's iterates, in order, as the run's iterates (see the module docstring)."""

    status: Status = "validated"
    #: Iterative, so that a ``[validation]`` split is scored at every iterate and the
    #: stopping rule (the import uses ``select``) picks the posterior.
    iterative: bool = True

    def __init__(self, iterates: Sequence[ErtIterate]) -> None:
        if not iterates:
            raise ValueError("ERT has no iterations to import")
        self.iterates = list(iterates)
        self._next = 0
        self._scoring: NDArray[np.float64] | None = None
        self._last: Record | None = None

    def _request(self) -> Step:
        iterate = self.iterates[self._next]
        return Step(
            requests=[
                ForecastRequest("ert", iterate.iteration, iterate.state, iterate.realizations)
            ],
            iteration=iterate.iteration,
        )

    def start(self, setup: RunSetup) -> Step:
        """Draw the scoring perturbations (as ES-MDA) and request ERT's prior."""
        n = int(max(i.realizations.max() for i in self.iterates)) + 1
        self._scoring = keep(
            setup.observations.values[:, None]
            + setup.data_error.perturbations(setup.streams["perturbations"], n)
        )
        return self._request()

    def advance(self, forecasts: Forecasts) -> Step:
        """Keep the forecast iterate; request ERT's next iterate, if any."""
        assert self._scoring is not None
        iterate = self.iterates[self._next]
        forecast = forecasts["ert"]
        ids = iterate.realizations[forecast.ok]
        record = Record(
            iteration=iterate.iteration,
            kind="iterate",
            state=keep(iterate.state, forecast.ok),
            predicted=forecast.predicted,
            member_ids=ids,
            scoring=keep(self._scoring[:, ids]),
        )
        self._last = record
        self._next += 1
        if self._next == len(self.iterates):
            return Step(requests=[], iteration=iterate.iteration, records=[record])
        step = self._request()
        return Step(requests=step.requests, iteration=step.iteration, records=[record])

    def export_state(self) -> AlgorithmState:
        """Return the replay's position (an import is not resumed; the driver checkpoints)."""
        assert self._scoring is not None
        return AlgorithmState(arrays={"scoring": self._scoring}, values={"next": self._next})

    def restore_state(self, setup: RunSetup, state: AlgorithmState) -> Step:
        """Continue the replay where it was checkpointed."""
        self._scoring = keep(np.asarray(state.arrays["scoring"], dtype=np.float64))
        self._next = int(state.values["next"])
        return self._request()

    def estimate(self) -> Estimate:
        """Return ERT's last imported iterate and its forecast."""
        assert self._last is not None
        return Estimate(self._last.state, self._last.predicted, self._last.member_ids)


def read_iterates(case_dir: Path) -> tuple[list[ErtIterate], dict[str, Path], dict[str, Any]]:
    """Read every ERT iteration of an exported case from its run paths.

    Returns the iterates, the stored cases by parameter digest, and the export's manifest.

    Raises
    ------
    ConfigError
        If ``case_dir`` is not an exported case, or ERT has not run it.
    """
    manifest_path = case_dir / "apex_export.json"
    if not manifest_path.exists():
        raise ConfigError(f"{case_dir} has no apex_export.json: export it with `apex ert export`")
    manifest: dict[str, Any] = json.loads(manifest_path.read_text())
    runs = case_dir / "runs"
    iterations = sorted(
        {int(p.name.split("-", 1)[1]) for p in runs.glob("realization-*/iter-*")}
    ) if runs.exists() else []  # fmt: skip
    if not iterations:
        raise ConfigError(f"{runs} has no ERT run paths: run ERT on the case first")
    config = composition.load_config(Path(manifest["config"]))
    layout = composition.build_layout(config)
    fields = {f["variable"]: (name, f["file"]) for name, f in manifest["fields"].items()}
    iterates, cases = [], {}
    for k in iterations:
        columns, ids = [], []
        for path in sorted(runs.glob(f"realization-*/iter-{k}"), key=_realization):
            state = {}
            for variable in layout.variables:
                if variable.name not in fields:
                    continue  # a scalar: read from the surface below
                name, file = fields[variable.name]
                if not (path / file).exists():
                    break
                state[variable.name] = read_grdecl(path / file, name)
            else:
                try:
                    state.update(read_scalar_state(manifest, path))
                except ValueError:
                    continue  # ERT did not get this far for this realization
                cases[state_digest(state)] = path / manifest["eclbase"]
                columns.append(layout.stack({n: v[:, None] for n, v in state.items()})[:, 0])
                ids.append(_realization(path))
        if columns:
            iterates.append(
                ErtIterate(k, np.ascontiguousarray(np.stack(columns, axis=-1)), np.array(ids))
            )
    return iterates, cases, manifest


def _realization(path: Path) -> int:
    return int(path.parent.name.split("-", 1)[1])


def import_case(case_dir: Path, run_dir: Path | None = None) -> HistoryMatchResult:
    """Replay an exported case's ERT iterations into an APEX run directory.

    ``run_dir`` defaults to ``<case_dir>/apex_run``. The run's configuration is the exported
    one, with ``select`` as the stopping rule of a ``[validation]`` split (every ERT iterate
    is imported; the posterior is the minimum-CV one) and a serial executor.

    Raises
    ------
    ConfigError
        If the case was not exported or not run, or ``run_dir`` already holds a run.
    """
    iterates, cases, manifest = read_iterates(case_dir)
    config = composition.load_config(Path(manifest["config"]))
    target = (run_dir or case_dir / "apex_run").resolve()
    if target.exists() and any(target.iterdir()):
        raise ConfigError(f"{target} already holds a run; remove it or choose another")
    run_section = config.run.model_copy(
        update={"output_dir": target, "name": f"{config.run.name}_ert"}
    )
    updates: dict[str, Any] = {"run": run_section, "execution": PluginSection(name="serial")}
    if config.validation is not None:
        updates["validation"] = config.validation.model_copy(
            update={"stopping": PluginSection(name="select")}
        )
    config = config.model_copy(update=updates)
    run = composition.build_run(config)
    # ERT's stored responses stand in for the simulator, run one at a time.
    run.workflow.simulator = StoredResponses(cases)
    run.workflow.executor = SerialExecutor()
    run = composition.Run(run.config, run.workflow, ErtReplay(iterates), run.run_dir, run.warnings)
    result = composition.start_built(run, prior=iterates[0].state)
    (target / "ert_import.json").write_text(
        json.dumps(
            {
                "source": "ert",
                "case": str(case_dir.resolve()),
                "iterations": [i.iteration for i in iterates],
                "realizations": [int(i.realizations.size) for i in iterates],
                "written_by": "apex ert import",
            },
            indent=2,
        )
        + "\n"
    )
    return result
