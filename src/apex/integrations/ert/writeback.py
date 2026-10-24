"""Write an APEX run of an ERT case into ERT's storage (write-back).

An ``APEX_HISTORY_MATCH`` run lives in APEX's run directory; this adds it to the case's
ERT storage (``ENSPATH``) as an experiment ``apex:<run name>`` with one ensemble per
iterate (``apex_iter_<k>``, each the child of the one before), so that ERT's GUI shows its
parameters, responses and misfits next to ERT's own experiments.

- **Parameters:** each ``FIELD`` of the case gets what the forward model read: the state
  variable it came from, or with a parameterization the property it maps to,
  for the members of the iterate (realization = APEX member id).
- **Responses:** each member's forecast at the case's observations (history-matching and
  held-out data), as ``summary`` and ``gen_data`` responses; other types (RFT,
  breakthrough, seismic) are not written back (``ert_storage.json`` names them), nor the
  full summary vectors, which are in the run's simulation folders while they are kept.
- **Again:** an experiment already written for the run is extended with the iterates it
  lacks (a resumed run), never rewritten.

:func:`member_responses` needs no ERT; :func:`write_back` takes an open ERT storage.
"""

import datetime as dt
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from apex.domain.observations import DataKey
from apex.domain.state import StateLayout

#: The experiment name prefix and the ensembles' names.
EXPERIMENT_PREFIX = "apex:"
ENSEMBLE_NAME = "apex_iter_{}"


@dataclass(frozen=True)
class WriteBackResult:
    """What :func:`write_back` added."""

    experiment: str
    written: list[int]
    already: list[int]


def member_responses(
    keys: Sequence[DataKey],
    predicted: NDArray[np.float64],
    types: Mapping[str, str] | None = None,
) -> tuple[dict[str, list[dict[str, list[Any]]]], set[str]]:
    """Return each member's response rows by ERT response type, and the types left out.

    ``predicted`` is ``(n_data, n_members)`` at ``keys``; ``types`` maps a response key to
    its ERT type (default: every key a summary). Summaries give ``response_key``, ``time``,
    ``values``; ``gen_data`` (time ``"report step, index"``, ERT's index string) gives
    ``response_key``, ``report_step``, ``index``, ``values``. Field data (keys with an
    ``index``) and other types are left out (named in the returned set).

    Raises
    ------
    ValueError
        If the shapes disagree, or a key's time does not fit its type.
    """
    if predicted.shape[0] != len(keys):
        raise ValueError(f"{predicted.shape[0]} forecast rows for {len(keys)} keys")
    types = types or {}
    by_type: dict[str, list[tuple[int, DataKey]]] = {}
    skipped = set()
    for i, key in enumerate(keys):
        kind = types.get(key.datatype, "summary")
        if key.index is not None or kind not in ("summary", "gen_data"):
            skipped.add(kind if key.index is None else "field data")
            continue
        by_type.setdefault(kind, []).append((i, key))
    out: dict[str, list[dict[str, list[Any]]]] = {}
    for kind, rows in by_type.items():
        if kind == "summary":
            columns: dict[str, list[Any]] = {
                "response_key": [k.datatype for _, k in rows],
                "time": [dt.datetime.fromisoformat(k.time) for _, k in rows],
            }
        else:
            steps, indices = zip(*(_gen_data_index(k.time) for _, k in rows), strict=True)
            columns = {
                "response_key": [k.datatype for _, k in rows],
                "report_step": list(steps), "index": list(indices),
            }  # fmt: skip
        out[kind] = [
            {**columns, "values": [float(predicted[i, j]) for i, _ in rows]}
            for j in range(predicted.shape[-1])
        ]
    return out, skipped


def _gen_data_index(time: str) -> tuple[int, int]:
    step, _, index = time.partition(",")
    try:
        return int(step), int(index)
    except ValueError as error:
        raise ValueError(f"GEN_DATA time {time!r} is not 'report step, index'") from error


def run_keys(run_dir: Path) -> tuple[list[DataKey], list[DataKey], list[DataKey]]:
    """Return the run's history-matching, CV and test keys, in its forecasts' order."""
    from apex.domain.data_split import resolve
    from apex.services.composition import analysis_observations, snapshot_config
    from apex.storage.run_directory import RunDirectory

    config = snapshot_config(RunDirectory(run_dir))
    observations = analysis_observations(run_dir)
    if config.validation is None:
        return list(observations.keys), [], []
    split = resolve(
        observations,
        cv=config.validation.cv,
        test=config.validation.test,
        history_matching=config.validation.history_matching,
    )
    return list(split.history_matching), list(split.cv), list(split.test)


def write_back(storage: Any, case_file: Path, run_dir: Path) -> WriteBackResult:  # noqa: ANN401
    """Add the run's iterates to ``storage`` (an ERT storage open for writing).

    Raises
    ------
    ValueError
        If the run is not an ``ert``-simulator run of this case, or has no results.
    """
    import polars as pl

    schemas: dict[str, Any] = {
        "summary": {"response_key": pl.String, "time": pl.Datetime("ms"), "values": pl.Float32},
        "gen_data": {
            "response_key": pl.String, "report_step": pl.UInt16, "index": pl.UInt16,
            "values": pl.Float32,
        },
    }  # fmt: skip
    from ert.storage.local_experiment import ExperimentType

    from apex.integrations.ert.config import parse_case
    from apex.integrations.ert.parameters import gen_kw_keys
    from apex.services.composition import (
        build_layout,
        build_parameterizations,
        load_results,
        snapshot_config,
    )
    from apex.storage.run_directory import RunDirectory
    from apex.workflows.parameterization import PropertyMap

    config = snapshot_config(RunDirectory(run_dir))
    if config.simulator.name != "ert":
        raise ValueError(f"{run_dir} is not an APEX_HISTORY_MATCH run (simulator ert)")
    fields: dict[str, dict[str, str]] = dict(config.simulator.options["fields"])
    step_parameters: dict[str, str] = dict(config.simulator.options.get("step_parameters") or {})
    case = parse_case(case_file)
    ensemble_config = case.ensemble_config
    name = EXPERIMENT_PREFIX + config.run.name
    experiment = next((e for e in storage.experiments if e.name == name), None)
    if experiment is None:
        experiment = storage.create_experiment(
            experiment_config={
                "parameter_configuration": [
                    p.model_dump(mode="json") for p in ensemble_config.parameter_configs.values()
                ],
                "response_configuration": [
                    r.model_dump(mode="json") for r in ensemble_config.response_configs.values()
                ],
                "observations": [o.model_dump(mode="json") for o in case.observation_declarations],
                "experiment_type": ExperimentType.MANUAL,
            },
            name=name,
        )
    existing = {e.iteration: e for e in experiment.ensembles}
    results = load_results(run_dir)
    layout: StateLayout = build_layout(config)
    properties = PropertyMap(build_parameterizations(config), layout)
    hm, cv, test = run_keys(run_dir)
    size = int(config.prior.ensemble_size or case.runpath_config.num_realizations)
    written, already = [], []
    not_written: set[str] = set()
    parent = None
    for k in results.accepted_iterations:
        if k in existing:
            already.append(k)
            parent = existing[k]
            continue
        iterate = results.iterate(k)
        ensemble = experiment.create_ensemble(
            ensemble_size=size, name=ENSEMBLE_NAME.format(k), iteration=k,
            prior_ensemble=parent,
        )  # fmt: skip
        realizations = np.asarray(iterate.member_ids, dtype=np.int64)
        # The case's FIELDs are what the forward model read: the properties, with a
        # parameterization, else the state itself.
        values = properties.ensemble(iterate.state)
        for name, field in fields.items():
            ensemble.save_parameters_numpy(np.asarray(values[name]), field["keyword"], realizations)
        for name, ert_name in step_parameters.items():
            block = np.asarray(values[name])
            keys = gen_kw_keys(case, ert_name)
            if keys:
                # ERT stores each GEN_KW key on its own, in its standard-normal space.
                for row, key in enumerate(keys):
                    ensemble.save_parameters_numpy(block[row : row + 1], key.name, realizations)
            else:
                ensemble.save_parameters_numpy(block, ert_name, realizations)
        blocks = [
            (hm, iterate.predicted),
            (cv, iterate.cv_predicted),
            (test, iterate.test_predicted),
        ]
        keys = [key for block_keys, values in blocks if values is not None for key in block_keys]
        predicted = np.vstack([values for _, values in blocks if values is not None])
        responses, skipped = member_responses(
            keys, predicted, experiment.response_key_to_response_type
        )
        not_written |= skipped
        for kind, per_member in responses.items():
            for member, rows in zip(realizations, per_member, strict=True):
                ensemble.save_response(kind, pl.DataFrame(rows, schema=schemas[kind]), int(member))
        written.append(k)
        parent = ensemble
    record = {
        "experiment": name, "experiment_id": str(experiment.id), "written": written,
        "already": already, "selected_iteration": results.selected_iteration,
        "posterior_iteration": results.posterior_iteration,
        "responses_not_written": sorted(not_written),
    }  # fmt: skip
    (run_dir / "ert_storage.json").write_text(json.dumps(record, indent=2) + "\n")
    return WriteBackResult(name, written, already)
