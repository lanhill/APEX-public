"""Upstream PET's distance localization with entries (``distance_entries``).

PET parity is in ``tests/parity/test_distance_entries.py``; these check the options, the
entry rules and the state round trip.
"""

from pathlib import Path
from typing import cast

import numpy as np
import pytest

from apex.algorithms.data_assimilation import (
    ApproxUpdate,
    DataErrorCovariance,
    EntryDistanceLocalization,
    GNEnRML,
    LMEnRML,
    LocalAnalysisUpdate,
    RLMMACUpdate,
    localizations,
)
from apex.algorithms.data_assimilation.entry_localization import Entry, planar_taper
from apex.domain import DataKey

GRID = [4, 3, 2]
CELLS = 24
KEYS = [DataKey("1", "WOPR:P1"), DataKey("1", "WBHP:I1"), DataKey("2", "WOPR:P1")]


def _make(entries: list[Entry]) -> EntryDistanceLocalization:
    return EntryDistanceLocalization(
        GRID, entries, variable_names=["PERMX", "PORO"], block_sizes=[CELLS, CELLS]
    )


def _prepared(localization: EntryDistanceLocalization) -> np.ndarray:
    localization.prepare(np.zeros((2 * CELLS, 5)), np.zeros((3, 5)), KEYS, np.random.default_rng())
    assert localization.taper_matrix is not None
    return localization.taper_matrix


def test_region_entries_mark_their_cell_and_layers() -> None:
    taper = _prepared(
        _make(
            [
                {
                    "data": "WOPR:P1",
                    "parameter": "PORO",
                    "taper": "region",
                    "position": [2, 3, 1],
                    "z_range": 0,
                }
            ]
        )
    )

    column = np.zeros(2 * CELLS)
    column[CELLS + (2 - 1) + 4 * (3 - 1)] = 1.0  # PORO, Eclipse order, layer 1 only
    np.testing.assert_array_equal(taper[:, 0], column)
    np.testing.assert_array_equal(taper[:, 2], column)  # time "*" (the default)
    assert not taper[:, 1].any()  # no entry for WBHP:I1: not updated


def test_a_later_entry_replaces_an_earlier_one() -> None:
    first: Entry = {"taper": "region", "position": [1, 1, 1]}
    second: Entry = {"time": "2", "parameter": "permx", "taper": "region", "position": [4, 3, 2]}

    taper = _prepared(_make([first, second]))

    assert taper[0, 0] == 1
    assert taper[0, 2] == 0  # PERMX at time 2 from the second entry
    assert taper[CELLS - 1, 2] == 1
    assert taper[CELLS, 2] == 1  # PORO still from the first


def test_the_planar_kernels() -> None:
    gc = planar_taper((5, 5), (2, 2), "gaspari_cohn", 1.0, (1.0, 0.0), 10)
    assert gc[2, 2] == 1
    assert gc[2, 4] == 0  # vanishes at 2 radius
    fb = planar_taper((5, 5), (2, 2), "furrer_bengtsson", 2.0, (1.0, 0.0), 10)
    assert fb[2, 2] == pytest.approx(10 / 12)
    assert fb[2, 4] == 0
    # A ratio of 2 divides the j offset by 2, doubling the reach along j.
    stretched = planar_taper((5, 5), (2, 2), "gaspari_cohn", 1.0, (2.0, 0.0), 10)
    assert stretched[4, 2] > 0
    assert stretched[2, 4] == 0


def test_the_import_kernel_reads_its_taper(tmp_path: Path) -> None:
    values = np.arange(CELLS, dtype=float).reshape(GRID, order="F") / CELLS
    np.save(tmp_path / "mask.npy", values)  # (nx, ny, nz): raveled with i fastest

    taper = _prepared(
        _make([{"parameter": "PERMX", "taper": "import", "taper_file": str(tmp_path / "mask.npy")}])
    )

    np.testing.assert_allclose(taper[:CELLS, 0], np.arange(CELLS) / CELLS)
    assert not taper[CELLS:].any()


def test_actnum_keeps_the_active_cells(tmp_path: Path) -> None:
    active = np.ones(CELLS, dtype=bool)
    active[:4] = False
    np.save(tmp_path / "actnum.npy", active)
    localization = EntryDistanceLocalization(
        GRID, [{"taper": "region", "position": [1, 2, 1]}], ["PERMX"], [20],
        actnum_file=tmp_path / "actnum.npy",
    )  # fmt: skip

    taper = _prepared_one(localization)

    assert taper.shape == (20, 3)
    assert taper[0, 0] == 1  # cell (1, 2, 1) is the 5th, the 1st active


def _prepared_one(localization: EntryDistanceLocalization) -> np.ndarray:
    localization.prepare(np.zeros((20, 5)), np.zeros((3, 5)), KEYS, np.random.default_rng())
    assert localization.taper_matrix is not None
    return localization.taper_matrix


@pytest.mark.parametrize(
    ("entry", "match"),
    [
        ({"taper": "gc", "position": [1, 1, 1], "radius": 1}, "taper must be"),
        ({"taper": "region", "position": [5, 1, 1]}, "inside the grid"),
        ({"taper": "gaspari_cohn", "position": [1, 1, 1]}, "radius"),
        ({"taper": "region", "position": [1, 1, 1], "parameter": "SWAT"}, "no state variable"),
        ({"taper": "region", "position": [1, 1, 1], "z_range": "some"}, "z_range"),
        ({"taper": "region", "position": [1, 1, 1], "colour": "red"}, "unknown keys"),
        ({"taper": "import"}, "needs a taper_file"),
    ],
)
def test_bad_entries_are_refused(entry: dict[str, object], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        _make([entry])  # type: ignore[list-item]


@pytest.mark.parametrize(
    ("entry", "match"),
    [({"data": "WOPR:P9"}, "no assimilated data 'WOPR:P9'"), ({"time": "3"}, "at time '3'")],
)
def test_entries_naming_missing_data_are_refused(entry: dict[str, object], match: str) -> None:
    localization = _make([cast(Entry, {"taper": "region", "position": [1, 1, 1], **entry})])
    with pytest.raises(ValueError, match=match):
        _prepared(localization)


def test_variables_off_the_grid_are_refused() -> None:
    with pytest.raises(ValueError, match="has 10 entries"):
        EntryDistanceLocalization(GRID, [{"taper": "region", "position": [1, 1, 1]}],
                                  ["PERMX", "MULT"], [CELLS, 10])  # fmt: skip
    # A variable no entry names may have any size (it is never updated by these data).
    only_permx: Entry = {"parameter": "PERMX", "taper": "region", "position": [1, 1, 1]}
    EntryDistanceLocalization(GRID, [only_permx], ["PERMX", "MULT"], [CELLS, 10])


def test_the_taper_is_rebuilt_from_its_exported_state() -> None:
    entries: list[Entry] = [{"taper": "furrer_bengtsson", "position": [2, 2, 1], "radius": 2}]
    original = _make(entries)
    taper = _prepared(original)

    restored = _make(entries)
    restored.restore_state(original.export_state())

    np.testing.assert_array_equal(restored.taper_matrix, taper)


def test_it_is_registered_and_localizes_enrml() -> None:
    assert localizations.factory("distance_entries") is EntryDistanceLocalization
    strategy = ApproxUpdate(localization=_make([{"taper": "region", "position": [1, 1, 1]}]))
    LMEnRML(strategy=strategy)
    GNEnRML(strategy=strategy)


@pytest.mark.parametrize("angle", [0.0, 30.0, 45.0, -60.0])
def test_the_kernel_is_centred(angle: float) -> None:
    """Point symmetry about the centre: what PET's asymmetric stencil crop breaks."""
    taper = planar_taper((21, 21), (10, 10), "gaspari_cohn", 7.0, (0.4, angle), 20)
    np.testing.assert_allclose(taper, taper[::-1, ::-1], atol=1e-15)


@pytest.mark.parametrize("shape", [(2, 3, 4), (4, 6), (24, 1)])
def test_files_in_another_layout_are_refused(tmp_path: Path, shape: tuple[int, ...]) -> None:
    np.save(tmp_path / "mask.npy", np.ones(shape))
    with pytest.raises(ValueError, match="flat in Eclipse order"):
        _make([{"taper": "import", "taper_file": str(tmp_path / "mask.npy")}])
    np.save(tmp_path / "actnum.npy", np.ones(shape, dtype=bool))
    with pytest.raises(ValueError, match="flat in Eclipse order"):
        EntryDistanceLocalization(
            GRID, [{"taper": "region", "position": [1, 1, 1]}], ["PERMX"], [CELLS],
            actnum_file=tmp_path / "actnum.npy",
        )  # fmt: skip


def test_an_npz_actnum_is_read_from_its_actnum_array(tmp_path: Path) -> None:
    active = np.ones(CELLS, dtype=bool)
    active[0] = False
    np.savez(tmp_path / "grid.npz", poro=np.ones(CELLS), actnum=active)
    localization = EntryDistanceLocalization(
        GRID, [{"taper": "region", "position": [2, 1, 1]}], ["PERMX"], [CELLS - 1],
        actnum_file=tmp_path / "grid.npz",
    )  # fmt: skip

    assert localization.n_cells == CELLS - 1


def test_rlmmac_and_local_analysis_take_it() -> None:
    rng = np.random.default_rng(1)
    state, predicted = rng.standard_normal((2 * CELLS, 8)), rng.standard_normal((3, 8))
    error = DataErrorCovariance.diagonal(np.full(3, 0.1))
    perturbed = predicted + rng.standard_normal((3, 8))
    entries: list[Entry] = [{"taper": "gaspari_cohn", "position": [2, 2, 1], "radius": 1.5}]
    for strategy in (
        RLMMACUpdate(localization=_make(entries)),
        LocalAnalysisUpdate(localization=_make(entries)),
    ):
        localization = strategy.localization
        assert localization is not None
        localization.prepare(state, predicted, KEYS, rng)
        reference = predicted.mean(axis=-1) if strategy.requires_reference_prediction else None
        step = strategy.step(
            state, predicted, perturbed, error, damping=1.0,
            reference_prediction=reference, rng=rng,
        )  # fmt: skip
        assert step.shape == state.shape
        assert np.all(
            step[np.flatnonzero(_make(entries).entry_taper(_make(entries).entries[0], 8) == 0)] == 0
        )
