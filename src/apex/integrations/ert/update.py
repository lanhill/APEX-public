"""APEX's analysis inside ERT's ES and ES-MDA (``APEX_UPDATE``).

ERT runs its experiment (sampling, forecasts, its update); the workflow job ``APEX_UPDATE``,
hooked at ``POST_UPDATE``, then replaces the posterior ERT just wrote with APEX's: one
ES-MDA analysis step (:class:`~apex.algorithms.data_assimilation.esmda.ESMDA`) on the
prior's parameters and responses, with ``apex.toml``'s strategy and localization, ERT's
weights as the schedule (step ``iteration``; ``(1,)`` for ES), and the un-inflated
observation errors.

This module holds the part that needs no ERT (:func:`apex_update`, on arrays, and
:func:`read_update_config`); :mod:`.plugin` reads ERT's storage and writes the posterior.

- **Random streams:** APEX draws its own perturbations, from ``[run] seed`` (default ERT's
  ``RANDOM_SEED``) and the step: ``SeedSequence(seed, spawn_key=(iteration, k))``, ``k = 0``
  for the perturbations and 1 for a localization's draws. ERT's perturbations are not
  APEX's, so ERT's discarded posterior and APEX's differ even for the same method.
- **Localizations prepared from the prior** (``adaptive``, ``distance``,
  ``distance_entries``) are prepared at the first update and stored (``prepared_<step>``),
  so later steps use the first prior's taper, as in an APEX run.
- **Bounds:** a field's ``MIN``/``MAX`` bound the state as APEX bounds it (clipped after the
  update); ERT applies them only to the files it writes.
"""

import tomllib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from pydantic import ValidationError

from apex.algorithms.data_assimilation.data_error import DataErrorCovariance
from apex.algorithms.data_assimilation.esmda import ESMDA
from apex.core.errors import ConfigError
from apex.domain.config import MethodSection
from apex.domain.observations import DataKey
from apex.domain.state import StateLayout, StateVariable

#: The methods whose one step ERT's schedule can run (iterative ones need B2).
ONE_STEP_METHODS = ("es", "esmda")
#: The sections ``apex.toml`` may have for ``APEX_UPDATE``.
UPDATE_SECTIONS = frozenset({"method", "run"})


@dataclass(frozen=True)
class UpdateConfig:
    """``apex.toml`` for ``APEX_UPDATE``: the strategy and localization, and the seed."""

    method: MethodSection
    seed: int | None


@dataclass(frozen=True)
class ErtUpdateInput:
    """One ERT update step's arrays, ensemble last.

    Attributes
    ----------
    parameters
        Each parameter group's prior, ``(n_values, n_realizations)``, in ERT's group order.
    bounds
        Each group's ``(lower, upper)`` (``-inf``/``inf`` when unbounded).
    predicted
        The prior's responses at the observations, ``(n_data, n_realizations)``.
    observations, errors
        Observed values and their un-inflated standard deviations, ``(n_data,)``.
    keys
        The data's keys (``DataKey(date, "VECTOR:WELL")``), in the rows' order.
    schedule
        ERT's inflation schedule (its normalized ES-MDA weights; ``(1,)`` for ES), so that
        APEX checks it as its own (``sum(1 / alpha) = 1``).
    iteration
        ERT's prior iteration: the step of the schedule (0 for the first update).
    """

    parameters: Mapping[str, NDArray[np.float64]]
    bounds: Mapping[str, tuple[float, float]]
    predicted: NDArray[np.float64]
    observations: NDArray[np.float64]
    errors: NDArray[np.float64]
    keys: Sequence[DataKey]
    schedule: tuple[float, ...]
    iteration: int

    @property
    def inflation(self) -> float:
        """The step's ``alpha``."""
        return self.schedule[self.iteration]


def read_update_config(path: Path) -> UpdateConfig:
    """Read and check ``apex.toml`` for ``APEX_UPDATE``.

    ``*_file`` options of the strategy and localization (and a ``distance_entries``
    entry's ``taper_file``) are resolved against the file's folder.

    Raises
    ------
    ConfigError
        If the file cannot be read, has other sections, names an iterative method, or
        sets an inflation (ERT's weights are the schedule).
    """
    try:
        raw: dict[str, Any] = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise ConfigError(f"cannot read {path}: {error}") from error
    extra = sorted(set(raw) - UPDATE_SECTIONS)
    if extra:
        raise ConfigError(
            f"{path}: APEX_UPDATE reads [method] and [run] seed only; not [{'], ['.join(extra)}]"
        )
    try:
        method = MethodSection.model_validate(raw.get("method", {"name": "esmda"}))
    except ValidationError as error:
        raise ConfigError(f"{path}: invalid [method]:\n{error}") from error
    if method.name not in ONE_STEP_METHODS:
        raise ConfigError(
            f"{path}: APEX_UPDATE runs one analysis step of ERT's schedule, so only "
            f"{' and '.join(ONE_STEP_METHODS)}; {method.name!r} is iterative: run it with "
            "APEX_HISTORY_MATCH"
        )
    if "inflation" in method.options:
        raise ConfigError(f"{path}: ERT's ES-MDA weights are the inflation; remove it")
    base = path.resolve().parent
    resolved = {
        name: part.model_copy(update={"options": _resolved(part.options, base)})
        for name, part in (("strategy", method.strategy), ("localization", method.localization))
        if part is not None
    }
    method = method.model_copy(update=resolved)
    run = dict(raw.get("run", {}))
    if set(run) - {"seed"}:
        raise ConfigError(f"{path}: APEX_UPDATE reads [run] seed only")
    return UpdateConfig(method, None if "seed" not in run else int(run["seed"]))


def _resolved(options: Mapping[str, Any], base: Path) -> dict[str, Any]:
    def resolve(value: str) -> str:
        return value if Path(value).is_absolute() else str(base / value)

    resolved: dict[str, Any] = {
        key: resolve(value) if key.endswith("_file") and isinstance(value, str) else value
        for key, value in options.items()
    }
    entries = resolved.get("entries")
    if isinstance(entries, list):
        resolved["entries"] = [
            {**e, "taper_file": resolve(str(e["taper_file"]))}
            if isinstance(e, dict) and "taper_file" in e
            else e
            for e in entries
        ]
    return resolved


def layout_of(update: ErtUpdateInput) -> StateLayout:
    """Return the state layout of the update's groups (variables named in lower case)."""
    return StateLayout([
        StateVariable(name.lower(), block.shape[0], *update.bounds[name])
        for name, block in update.parameters.items()
    ])  # fmt: skip


def streams(seed: int, iteration: int) -> tuple[np.random.Generator, np.random.Generator]:
    """Return the step's perturbation and localization generators."""
    perturbation, localization = (
        np.random.default_rng(np.random.SeedSequence(seed, spawn_key=(iteration, k)))
        for k in (0, 1)
    )
    return perturbation, localization


def apex_update(
    update: ErtUpdateInput,
    config: UpdateConfig,
    seed: int,
    prepared: Mapping[str, NDArray[np.float64]] | None = None,
) -> tuple[dict[str, NDArray[np.float64]], dict[str, NDArray[np.float64]]]:
    """Run APEX's analysis step on one ERT update's arrays.

    Parameters
    ----------
    update
        The step's arrays.
    config
        ``apex.toml``'s method (its strategy and localization).
    seed
        The root seed of the step's streams (:func:`streams`).
    prepared
        A localization's arrays prepared at an earlier step (``None`` at the first).

    Returns
    -------
    tuple
        The posterior per group ``(n_values, n_realizations)`` and the localization's
        prepared arrays (to store for the next step).

    Raises
    ------
    ConfigError
        If the method cannot be built for these parameters.
    ValueError
        If the arrays do not agree.
    """
    from apex.services.composition import build_method

    layout = layout_of(update)
    section = config.method.model_copy(
        update={"name": "esmda", "options": {"inflation": [float(a) for a in update.schedule]}}
    )
    method = build_method(section, layout)
    assert isinstance(method, ESMDA)  # the section names esmda
    state = np.ascontiguousarray(
        np.vstack([np.asarray(b, dtype=np.float64) for b in update.parameters.values()])
    )
    predicted = np.ascontiguousarray(update.predicted, dtype=np.float64)
    if predicted.shape != (len(update.keys), state.shape[-1]):
        raise ValueError(
            f"responses {predicted.shape} do not match {len(update.keys)} data and "
            f"{state.shape[-1]} realizations"
        )
    perturbation_rng, localization_rng = streams(seed, update.iteration)
    if prepared:
        method.restore_prepared(prepared)
    else:
        method.prepare(state, predicted, list(update.keys), localization_rng)
    data_error = DataErrorCovariance.diagonal(np.asarray(update.errors, dtype=np.float64) ** 2)
    observations = np.asarray(update.observations, dtype=np.float64)
    result = method.analysis(
        update.iteration, state, predicted, observations, data_error, perturbation_rng,
        localization_rng=localization_rng,
    )  # fmt: skip
    blocks = layout.split(result.posterior)
    posterior = {name: blocks[name.lower()] for name in update.parameters}
    return posterior, dict(method.export_prepared())
