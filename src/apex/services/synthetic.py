"""A created synthetic case's truth and observations.

- :func:`draw_truth`: one realization of the case's own prior, from a seed of its own (not
  the members' stream), clipped to the bounds as members are: the truth is then a plausible
  member, as synthetic studies assume.
- :func:`observe_truth`: the truth run once on the case's simulator, through the process
  executor (the machine-wide pools, such as Eclipse's licences, and its transient retries),
  its wells' vectors read at the report dates, and Gaussian noise of the data errors added
  (seeded, recorded). The one run is the only simulation a created case needs; callers ask
  before it (the CLI's prompt, ``--yes``). A failed run's folder is kept for its logs.

A data error's floor is a fraction of its well's scale: of the largest rate any of the
well's rate vectors reaches, and of its largest pressure for pressures. A water rate of 0
before breakthrough is observed, with the floor of its well's rates: it tells which
members break through too early. Only a well whose every rate is 0 is left out.
"""

import csv
import datetime as dt
import io
import shutil
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from apex.core.errors import ConfigError
from apex.core.plugins import executors, priors, simulators
from apex.core.simulation import MemberFailure, MemberInput, PriorGenerator
from apex.domain.observations import DataKey
from apex.domain.state import StateVariable

#: The truth's member id in its work folder's name.
TRUTH_ID = 0


def draw_truth(config: Mapping[str, Any], seed: int) -> dict[str, NDArray[np.float64]]:
    """Return one realization of ``config``'s prior per state variable, from ``seed``.

    Only the per-variable generators of ``[prior.variables]`` (a created case's own).

    Raises
    ------
    ConfigError
        If a variable has no generator there, or the generator refuses its options.
    """
    import apex.services.composition  # noqa: F401  (registers the prior generators)

    rng = np.random.default_rng(seed)
    sections = dict(dict(config.get("prior", {})).get("variables", {}))
    truth = {}
    for raw in dict(config.get("state", {})).get("variables", []):
        variable = StateVariable(name=str(raw["name"]), size=int(raw["size"]),
                                 lower=float(raw.get("lower", -np.inf)),
                                 upper=float(raw.get("upper", np.inf)),
                                 grid=tuple(raw["grid"]) if raw.get("grid") else None)  # fmt: skip
        section = sections.get(variable.name)
        if section is None:
            raise ConfigError(f"{variable.name} has no generator in [prior.variables]")
        try:
            generator = priors.create(str(section["name"]), **dict(section.get("options", {})))
            if not isinstance(generator, PriorGenerator):
                raise ConfigError(f"the prior of {variable.name} is drawn jointly: not supported")
            # Generators draw ensembles of two or more: the truth is the first of two.
            block = generator.generate(variable, 2, rng)
        except (TypeError, ValueError, ImportError) as error:
            raise ConfigError(f"the prior of {variable.name}: {error}") from error
        truth[variable.name] = np.clip(block[:, 0], variable.lower, variable.upper)
    return truth


@dataclass
class Observed:
    """The observations written, and what was left out."""

    rows: int
    vectors: list[str]
    left_out: list[str] = field(default_factory=list)


def observe_truth(
    *,
    simulator: str,
    options: Mapping[str, Any],
    truth: Mapping[str, NDArray[np.float64]],
    vectors: Sequence[str],
    dates: Sequence[dt.date],
    relative_error: float,
    floor_fraction: float,
    seed: int,
    work_dir: Path,
    output: Path,
    executor: str = "process",
) -> Observed:
    """Run the truth once and write its noisy observations to ``output`` (APEX's CSV).

    Each value's standard deviation is ``max(relative_error * |value|, floor)`` with
    ``floor = floor_fraction * the vector's largest |value|``; the noise is drawn from
    ``seed``. ``work_dir`` holds the run, removed after a success. ``executor`` runs it
    (``process``: the machine's pools and retries; tests use ``serial``).

    Raises
    ------
    ConfigError
        If the simulator cannot be made, or the run fails.
    """
    import apex.services.composition  # noqa: F401  (registers the simulators and executors)

    keys = [DataKey(date.isoformat(), vector) for date in dates for vector in vectors]
    try:
        adapter = simulators.create(simulator, **dict(options))
        runner = executors.create(executor, **({"workers": 1} if executor == "process" else {}))
    except (TypeError, ValueError) as error:
        raise ConfigError(f"the simulator {simulator!r}: {error}") from error
    work_dir.mkdir(parents=True, exist_ok=True)
    member = MemberInput(TRUTH_ID, {k: np.ascontiguousarray(v) for k, v in truth.items()}, work_dir)
    (result,) = runner.run_all(adapter, [member], keys)
    if isinstance(result, MemberFailure):
        raise ConfigError(f"the truth run failed (its files are in {work_dir.name}/): "
                          f"{result.message}")  # fmt: skip
    shutil.rmtree(work_dir, ignore_errors=True)
    values = np.asarray(result)
    scales = _scales(keys, values)
    rng = np.random.default_rng(seed)
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(["time", "datatype", "value", "variance"])
    observed: list[str] = []
    left_out: list[str] = []
    rows = 0
    for key, value in zip(keys, values, strict=True):
        scale = scales[_scale_key(key.datatype)]
        if scale == 0:
            if key.datatype not in left_out:
                left_out.append(key.datatype)
            continue
        if key.datatype not in observed:
            observed.append(key.datatype)
        sigma = max(relative_error * abs(float(value)), floor_fraction * scale)
        noisy = float(value) + sigma * float(rng.standard_normal())
        writer.writerow([key.time, key.datatype, repr(noisy), repr(sigma * sigma)])
        rows += 1
    if not rows:
        raise ConfigError("the truth run gave no non-zero data to observe")
    output.write_text(buffer.getvalue())
    return Observed(rows, observed, left_out)


#: Pressure vectors: their floor is a fraction of the well's pressure, not of its rates.
PRESSURES = ("WBHP", "WTHP")


def _scale_key(datatype: str) -> tuple[str, str]:
    vector, _, well = datatype.partition(":")
    return well, "pressure" if vector in PRESSURES else "rate"


def _scales(keys: Sequence[DataKey], values: NDArray[np.float64]) -> dict[tuple[str, str], float]:
    """Return each well's scale: its largest |rate| over every rate vector, or |pressure|."""
    scales: dict[tuple[str, str], float] = {}
    for key, value in zip(keys, values, strict=True):
        which = _scale_key(key.datatype)
        scales[which] = max(scales.get(which, 0.0), abs(float(value)))
    return scales
