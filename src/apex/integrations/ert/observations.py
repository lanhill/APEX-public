r"""Every ERT observation type for ``APEX_HISTORY_MATCH``.

ERT (25 and 26) has summary, general (``GEN_DATA``), RFT, breakthrough and seismic observations,
each with its own response type and its own way of matching a response to an observation.
APEX takes them all through ERT's own code:

- **The case's observations** (:func:`case_observations`): ERT builds its observation
  tables from the case's declarations (in a scratch storage), and each row's key is ERT's
  own index string of the row (``index_column_expr``: ``"0, 42"`` for ``GEN_DATA``'s report
  step and index, a date for summaries).
- **A batch's responses** (:func:`batch_responses`, and this module as a program for the
  ``ert`` executor): after ``ert ensemble_experiment``, ERT has loaded every response into
  the batch's storage; ``get_observations_and_responses`` matches them to the observations
  as ERT's own update does, row by row with the same index strings.

APEX's key of a row (:func:`apex_key`) is ``DataKey(date, "VECTOR:WELL")`` for a summary
(as in summary-only cases) and ``DataKey(index, response_key)`` otherwise. ERT stores
observations and responses as float32; summary observations keep their declared values.

    python -m apex.integrations.ert.observations --storage DIR --keys KEYS.json \
        --realizations N --out RESPONSES.npz
"""

import argparse
import json
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from apex.domain.observations import DataKey

#: The response type whose rows keep APEX's summary keys.
SUMMARY = "summary"


def apex_key(response_type: str, response_key: str, index: str) -> DataKey:
    """Return APEX's key of one ERT observation row."""
    if response_type == SUMMARY:
        # "1970-01-31 00:00:00.000": APEX's summary keys are dates.
        date, _, clock = index.partition(" ")
        return DataKey(date if clock.strip("0:.") == "" else index, response_key)
    return DataKey(index, response_key)


def case_observations(config: Any) -> list[tuple[str, DataKey, float, float]]:  # noqa: ANN401
    """Return ``(type, key, value, error)`` of every observation of a parsed ERT case."""
    from ert.storage import open_storage
    from ert.storage.local_experiment import ExperimentType

    ensemble_config = config.ensemble_config
    rows = []
    with tempfile.TemporaryDirectory() as scratch, open_storage(scratch, mode="w") as storage:
        experiment = storage.create_experiment(
            experiment_config={
                "parameter_configuration": [],
                "response_configuration": [
                    r.model_dump(mode="json") for r in ensemble_config.response_configs.values()
                ],
                "observations": [
                    o.model_dump(mode="json") for o in config.observation_declarations
                ],
                "experiment_type": ExperimentType.MANUAL,
            },
            name="apex-observations",
        )
        for response_type, frame in experiment.observations.items():
            response = experiment.response_configuration[response_type]
            indexed = frame.with_columns(response.index_column_expr().alias("__index__"))
            for key, index, value, error in zip(
                indexed["response_key"], indexed["__index__"], indexed["observations"],
                indexed["std"], strict=True,
            ):  # fmt: skip
                rows.append(
                    (response_type, apex_key(response_type, str(key), str(index)),
                     float(value), float(error))
                )  # fmt: skip
    return rows


def batch_responses(
    storage_path: Path, keys: Sequence[DataKey], n_realizations: int
) -> NDArray[np.float64]:
    """Return the responses of a batch at ``keys``, ``(n_keys, n_realizations)``.

    A realization without a response at a key (failed, or ERT's quality check) is NaN.

    Raises
    ------
    ValueError
        If the batch has no ensemble, or a key matches no observation of the batch.
    """
    from ert.storage import open_storage

    out = np.full((len(keys), n_realizations), np.nan)
    with open_storage(storage_path, mode="r") as storage:
        ensembles = [e for x in storage.experiments for e in x.ensembles]
        if not ensembles:
            raise ValueError(f"{storage_path} has no ensemble")
        (ensemble,) = ensembles
        experiment = ensemble.experiment
        realizations = np.flatnonzero(ensemble.get_realization_mask_with_responses())
        if realizations.size == 0:
            return out
        frame = ensemble.get_observations_and_responses(experiment.observation_keys, realizations)
        types = experiment.response_key_to_response_type
        pairs = zip(frame["response_key"], frame["index"], strict=True)
        rows = {apex_key(types[str(k)], str(k), str(i)): n for n, (k, i) in enumerate(pairs)}
        missing = [str(k) for k in keys if k not in rows]
        if missing:
            raise ValueError(f"no observation of the batch at {', '.join(missing[:5])}")
        positions = [rows[k] for k in keys]
        for real in realizations:
            values = frame[str(real)].to_numpy().astype(np.float64)
            out[:, real] = values[positions]
    return out


def main(argv: list[str] | None = None) -> int:
    """Write a batch's responses (see the module docstring)."""
    parser = argparse.ArgumentParser(description="Read an ERT batch's responses for APEX.")
    parser.add_argument("--storage", type=Path, required=True)
    parser.add_argument("--keys", type=Path, required=True)
    parser.add_argument("--realizations", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    keys = [DataKey(str(t), str(d)) for t, d in json.loads(args.keys.read_text())]
    values = batch_responses(args.storage, keys, args.realizations)
    np.save(args.out, values)
    return 0


if __name__ == "__main__":
    sys.exit(main())
