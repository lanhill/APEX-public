"""APEX's analysis inside ERT's ES-MDA, on recorded arrays, without ERT.

Reference: APEX's own ES-MDA step (``ESMDA.analysis``) on the same prior, responses and
streams; ``APEX_UPDATE`` must give exactly its posterior, so that an update in ERT is an
APEX update.
"""

import math
from pathlib import Path

import numpy as np
import pytest

from apex.algorithms.data_assimilation.data_error import DataErrorCovariance
from apex.core.errors import ConfigError
from apex.domain.observations import DataKey
from apex.integrations.ert.update import (
    ErtUpdateInput,
    apex_update,
    layout_of,
    read_update_config,
    streams,
)
from apex.services.composition import build_method

N, N_DATA = 30, 8


def _input(iteration: int = 0, schedule: tuple[float, ...] = (2.0, 2.0)) -> ErtUpdateInput:
    rng = np.random.default_rng(3)
    permx = rng.normal(5.0, 1.0, (12, N))
    poro = rng.uniform(0.1, 0.3, (12, N))
    forward = rng.standard_normal((N_DATA, 24))
    predicted = forward @ np.vstack([permx, poro])
    keys = [DataKey(f"2020-0{1 + t}-01", f"WOPR:P{w}") for w in (1, 2) for t in range(4)]
    return ErtUpdateInput(
        parameters={"PERMX": permx, "PORO": poro},
        bounds={"PERMX": (-math.inf, math.inf), "PORO": (0.05, 0.4)},
        predicted=predicted,
        observations=predicted.mean(axis=1) + 0.5,
        errors=np.full(N_DATA, 0.8),
        keys=keys,
        schedule=schedule,
        iteration=iteration,
    )


def _apex_toml(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "apex.toml"
    path.write_text(text)
    return path


APPROX = '[method]\nname = "esmda"\nstrategy = { name = "approx", options = { energy = 0.99 } }\n'


def _reference(
    update: ErtUpdateInput, config_text: str, tmp_path: Path, seed: int
) -> dict[str, np.ndarray]:
    """APEX's ES-MDA step on the same arrays, streams and schedule."""
    config = read_update_config(_apex_toml(tmp_path, config_text))
    section = config.method.model_copy(
        update={"name": "esmda", "options": {"inflation": list(update.schedule)}}
    )
    layout = layout_of(update)
    method = build_method(section, layout)
    state = np.vstack(list(update.parameters.values()))
    perturbation, localization = streams(seed, update.iteration)
    method.prepare(state, update.predicted, list(update.keys), localization)  # type: ignore[attr-defined]
    result = method.analysis(  # type: ignore[attr-defined]
        update.iteration, state, update.predicted, update.observations,
        DataErrorCovariance.diagonal(update.errors**2), perturbation,
        localization_rng=localization,
    )  # fmt: skip
    blocks = layout.split(result.posterior)
    return {"PERMX": blocks["permx"], "PORO": blocks["poro"]}


@pytest.mark.parametrize(
    "localization",
    [
        "",
        'localization = { name = "adaptive_prior" }\n',
        'localization = { name = "adaptive_projected" }\n',
    ],
)
def test_the_update_is_apexs_es_mda_step(tmp_path: Path, localization: str) -> None:
    text = APPROX + localization
    update = _input(iteration=1)

    posterior, _ = apex_update(update, read_update_config(_apex_toml(tmp_path, text)), seed=7)

    expected = _reference(update, text, tmp_path, seed=7)
    for group in ("PERMX", "PORO"):
        np.testing.assert_array_equal(posterior[group], expected[group])
    assert posterior["PORO"].min() >= 0.05  # the field's MIN/MAX bound the state
    assert posterior["PORO"].max() <= 0.4
    assert not np.array_equal(posterior["PERMX"], update.parameters["PERMX"])


def test_a_prior_localization_keeps_the_first_updates_taper(tmp_path: Path) -> None:
    config = read_update_config(
        _apex_toml(tmp_path, APPROX + 'localization = { name = "adaptive_prior" }\n')
    )
    first, prepared = apex_update(_input(iteration=0), config, seed=7)
    later = _input(iteration=1)
    moved = ErtUpdateInput(**{**later.__dict__, "parameters": first})

    with_first, _ = apex_update(moved, config, seed=7, prepared=prepared)
    with_own, _ = apex_update(moved, config, seed=7)

    assert set(prepared) == {"taper"}
    assert not np.array_equal(with_first["PERMX"], with_own["PERMX"])  # the taper is kept
    assert all(np.isfinite(v).all() for v in with_first.values())


def test_streams_are_per_step_and_reproducible(tmp_path: Path) -> None:
    config = read_update_config(_apex_toml(tmp_path, APPROX))

    once, _ = apex_update(_input(iteration=0), config, seed=7)
    again, _ = apex_update(_input(iteration=0), config, seed=7)
    other, _ = apex_update(_input(iteration=0), config, seed=8)

    np.testing.assert_array_equal(once["PERMX"], again["PERMX"])
    assert not np.array_equal(once["PERMX"], other["PERMX"])


def test_erts_schedule_is_checked_as_apexs(tmp_path: Path) -> None:
    config = read_update_config(_apex_toml(tmp_path, APPROX))

    with pytest.raises(ValueError, match=r"sum\(1/alpha\)"):
        apex_update(_input(schedule=(2.0, 4.0)), config, seed=7)
    single, _ = apex_update(_input(schedule=(1.0,)), config, seed=7)  # ERT's ES
    assert single["PERMX"].shape == (12, N)


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ('[method]\nname = "gies"\n', "APEX_HISTORY_MATCH"),
        (
            '[method]\nname = "esmda"\noptions = { inflation = [4, 4, 4, 4] }\n',
            "ERT's ES-MDA weights",
        ),
        (APPROX + "[prior]\nensemble_size = 3\n", r"not \[prior\]"),
        (APPROX + '[run]\nname = "x"\n', r"\[run\] seed only"),
    ],
)
def test_what_apex_update_cannot_do_is_refused(tmp_path: Path, text: str, message: str) -> None:
    with pytest.raises(ConfigError, match=message):
        read_update_config(_apex_toml(tmp_path, text))


def test_files_named_in_apex_toml_are_relative_to_it(tmp_path: Path) -> None:
    text = (
        '[method]\nname = "esmda"\n[method.strategy]\nname = "approx"\n'
        '[method.localization]\nname = "distance_entries"\n[method.localization.options]\n'
        'grid = [4, 3, 1]\nactnum_file = "actnum.npz"\nentries = [{ taper = "import", '
        'taper_file = "taper.npy" }]\n[run]\nseed = 5\n'
    )

    config = read_update_config(_apex_toml(tmp_path, text))

    assert config.seed == 5
    assert config.method.localization is not None
    options = config.method.localization.options
    assert options["actnum_file"] == str(tmp_path / "actnum.npz")
    assert options["entries"][0]["taper_file"] == str(tmp_path / "taper.npy")
