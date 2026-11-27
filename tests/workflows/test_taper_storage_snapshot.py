"""Runs started before the low-memory taper (snapshot version 7) resume with the dense one.

``adaptive``/``adaptive_prior`` keep their taper packed or recomputed and form the gain in
blocks by default since 2026-11-27, which agrees with the old dense computation up to
rounding only; an older snapshot is pinned to ``taper_storage = "dense"``,
``block_rows = 0`` so that its resume stays identical.
"""

import json
from typing import Any

import numpy as np
import pytest

from apex.services import composition
from apex.storage.run_directory import SNAPSHOT_VERSION, RunDirectory
from apex.testing import CaseFactory

DENSE = {"taper_storage": "dense", "block_rows": 0}


@pytest.mark.parametrize(
    ("method", "expected"),
    [
        (  # the method's localization, as a table and by its bare name
            {"name": "esmda", "localization": {"name": "adaptive_prior", "options": {}}},
            {"name": "esmda", "localization": {"name": "adaptive_prior", "options": DENSE}},
        ),
        (
            {"name": "gies", "localization": "adaptive"},
            {"name": "gies", "localization": {"name": "adaptive", "options": DENSE}},
        ),
        (  # the user's own options win; the others are pinned
            {"name": "gies", "localization": {"name": "adaptive", "options": {"block_rows": 9}}},
            {
                "name": "gies",
                "localization": {
                    "name": "adaptive",
                    "options": {"taper_storage": "dense", "block_rows": 9},
                },
            },
        ),
        (  # CHOP's hyper-parameter localization
            {"name": "chop", "options": {"hyper_localization": {"name": "adaptive_prior"}}},
            {
                "name": "chop",
                "options": {"hyper_localization": {"name": "adaptive_prior", "options": DENSE}},
            },
        ),
        (  # other localizations are not touched
            {"name": "esmda", "localization": {"name": "adaptive_per_update"}},
            {"name": "esmda", "localization": {"name": "adaptive_per_update"}},
        ),
        ({"name": "esmda"}, {"name": "esmda"}),
    ],
)
def test_old_snapshots_keep_the_dense_taper(
    method: dict[str, Any], expected: dict[str, Any]
) -> None:
    raw = {"method": json.loads(json.dumps(method))}

    composition._migrate_taper_storage(raw)

    assert raw["method"] == expected


def test_a_version_6_run_is_read_with_the_dense_taper(linear_case: CaseFactory) -> None:
    method = (
        'name = "esmda"\noptions = { inflation = [2, 2] }\n'
        'strategy = { name = "approx", options = { energy = 1.0 } }\n'
        'localization = { name = "adaptive_prior", options = { taper = "soft" } }'
    )
    case = linear_case(name="old", method=method, ensemble_size=20)
    run_dir = case.directory / "runs" / "old"
    composition.start(case.config, max_steps=0)
    assert SNAPSHOT_VERSION >= 7
    new = composition.snapshot_config(RunDirectory(run_dir)).method.localization
    assert new is not None
    assert "taper_storage" not in new.options  # the new defaults
    (run_dir / "snapshot_version").write_text("6\n")

    old = composition.snapshot_config(RunDirectory(run_dir)).method.localization

    assert old is not None
    assert old.options == {"taper": "soft", **DENSE}


def test_an_older_run_resumes_as_the_dense_computation(linear_case: CaseFactory) -> None:
    """End to end: a run that an older APEX started (the dense taper, no options in its
    snapshot, version 6) resumes exactly as a run of the dense computation."""
    dense = (
        'name = "esmda"\noptions = { inflation = [2, 2] }\n'
        'strategy = { name = "approx", options = { energy = 1.0 } }\n'
        'localization = { name = "adaptive_prior", options = { taper = "soft", '
        'taper_storage = "dense", block_rows = 0 } }'
    )
    reference = composition.start(linear_case(name="ref", method=dense, ensemble_size=30).config)
    case = linear_case(name="old", method=dense, ensemble_size=30)
    run_dir = case.directory / "runs" / "old"
    composition.start(case.config, max_steps=1)
    # What an older APEX wrote: no storage options in the snapshot, version 6.
    snapshot = json.loads((run_dir / "config.json").read_text())
    options = snapshot["method"]["localization"]["options"]
    del options["taper_storage"], options["block_rows"]
    (run_dir / "config.json").write_text(json.dumps(snapshot))
    (run_dir / "snapshot_version").write_text("6\n")

    resumed = composition.resume(run_dir)

    np.testing.assert_array_equal(resumed.posterior, reference.posterior)
    assert resumed.history == reference.history
