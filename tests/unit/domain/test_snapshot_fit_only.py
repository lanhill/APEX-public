"""Fit-only responses of a snapshot set.

Responses every run also returned for a plugin's fit to read (``[case] fit_only``): not the
surrogate's keys, so neither predicted nor gated; kept with the runs (``columns``), in the
digest only when present (older sets keep theirs), and stored with the set.
"""

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from apex.domain.observations import DataKey
from apex.domain.state import StateLayout, StateVariable
from apex.domain.surrogate import SnapshotProvenance, SnapshotSet, SurrogateCaseSection
from apex.storage.surrogate_directory import SurrogateDirectory, load_snapshots

KEYS = (DataKey("1", "WOPR:P"), DataKey("2", "WOPR:P"))
FIT_ONLY = (DataKey("1", "FPR"), DataKey("2", "FPR"))


def _set(fit_only: bool = True) -> SnapshotSet:
    return SnapshotSet(
        layout=StateLayout([StateVariable("x", 2)]), inputs={"x": np.arange(6.0).reshape(2, 3)},
        keys=KEYS, outputs=np.ones((2, 3)), run_ids=(0, 1, 2),
        provenance=SnapshotProvenance(simulator="test", plan="prior"),
        fit_only_keys=FIT_ONLY if fit_only else (),
        fit_only_outputs=np.array([[200.0, 201.0, 202.0], [190.0, 191.0, 192.0]])
        if fit_only else None,
    )  # fmt: skip


def test_fit_only_responses_follow_the_runs_and_the_digest() -> None:
    full = _set()
    part = full.columns([2, 0])
    np.testing.assert_array_equal(part.fit_only_outputs, [[202.0, 200.0], [192.0, 190.0]])
    assert part.fit_only_keys == FIT_ONLY
    plain = _set(fit_only=False)
    assert plain.digest() != full.digest()
    assert plain.columns([0, 1, 2]).digest() == plain.digest()


def test_fit_only_responses_are_checked() -> None:
    with pytest.raises(ValueError, match="come together"):
        SnapshotSet(layout=StateLayout([StateVariable("x", 2)]), inputs={"x": np.ones((2, 3))},
                    keys=KEYS, outputs=np.ones((2, 3)), run_ids=(0, 1, 2),
                    provenance=SnapshotProvenance(simulator="t", plan="prior"),
                    fit_only_keys=FIT_ONLY)  # fmt: skip
    with pytest.raises(ValueError, match="not the set's keys"):
        SnapshotSet(layout=StateLayout([StateVariable("x", 2)]), inputs={"x": np.ones((2, 3))},
                    keys=KEYS, outputs=np.ones((2, 3)), run_ids=(0, 1, 2),
                    provenance=SnapshotProvenance(simulator="t", plan="prior"),
                    fit_only_keys=KEYS[:1], fit_only_outputs=np.ones((1, 3)))  # fmt: skip


def test_fit_only_responses_are_stored_with_the_set(tmp_path: Path) -> None:
    full = _set()
    path = SurrogateDirectory(tmp_path).save_snapshots(full)
    again = load_snapshots(path)
    assert again.fit_only_keys == FIT_ONLY
    np.testing.assert_array_equal(again.fit_only_outputs, full.fit_only_outputs)
    assert again.digest() == full.digest()


def test_the_case_section_takes_fit_only_series() -> None:
    section = SurrogateCaseSection.model_validate(
        {"config_file": "case.toml", "fit_only": {"vectors": ["FPR"], "times": ["2021-02-01"]}}
    )
    assert section.fit_only is not None
    assert section.fit_only.vectors == ["FPR"]


def test_only_the_builds_fine_runs_ask_for_fit_only_keys() -> None:
    """A build's case lists the fit-only keys after its own (``run_keys``), and its snapshot
    set splits them off; a check's case (``surrogate_case``, as ``check_surrogate`` builds
    it) has none, and coarse runs ask for ``case.keys`` alone."""
    import inspect
    from types import SimpleNamespace

    from apex.services.surrogates import SurrogateCase, _snapshot_set, surrogate_case

    view = SimpleNamespace(keys=KEYS, fields=(), field_times=(), field_cells=0,
                           fit_only_keys=FIT_ONLY)  # fmt: skip
    assert SurrogateCase.run_keys.fget(view) == (*KEYS, *FIT_ONLY)  # type: ignore[attr-defined]
    assert "fit_only" not in str(inspect.signature(surrogate_case))
    assert SurrogateCase.__dataclass_fields__["fit_only_keys"].default == ()
    case: Any = SimpleNamespace(
        keys=KEYS, fit_only_keys=FIT_ONLY, run_keys=(*KEYS, *FIT_ONLY), fields=(),
        field_times=(), field_cells=0, gradients=False,
        input_layout=StateLayout([StateVariable("x", 1)]),
        properties=SimpleNamespace(ensemble=lambda state: state),
    )  # fmt: skip
    values = {m: np.array([1.0, 2.0, 200.0 + m, 190.0 + m]) for m in range(3)}
    got = _snapshot_set(case, [0, 1, 2], {"x": np.zeros((1, 3))}, values,
                        SnapshotProvenance(simulator="t", plan="prior"), [])  # fmt: skip
    np.testing.assert_array_equal(got.outputs, [[1.0] * 3, [2.0] * 3])
    np.testing.assert_array_equal(got.fit_only_outputs, [[200.0, 201.0, 202.0],
                                                          [190.0, 191.0, 192.0]])  # fmt: skip
