"""The low-memory forms of the prior adaptive localization (``taper_storage``, ``block_rows``).

Reference: the original computation, ``taper_storage = "dense"`` with ``block_rows = 0``,
which forms the whole ``(n_state, n_data)`` taper, gain and product. Every other form
computes the same quantities row by row; BLAS may round a product of fewer rows
differently (measured on the workstation: soft tapers within 4.4e-15, steps within 4.5e-15 of the
largest entry, hard tapers equal), so they must agree to ``TOL`` (1e-12, relative to the
largest entry), draw the same numbers from the localization's stream, and resume from
their own checkpoint bit for bit.
"""

import tracemalloc
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from apex.algorithms.data_assimilation import localization as module
from apex.algorithms.data_assimilation.local_analysis import LocalAnalysisUpdate
from apex.algorithms.data_assimilation.localization import (
    PriorAdaptiveLocalization,
    pack_taper,
)
from apex.domain.observations import DataKey

SIZES = [37, 23, 1, 9]  # a one-row variable takes the universal threshold's size-1 branch
NAMES = ["perm", "poro", "aq", "nn"]
N_DATA, N_ENS = 13, 20
#: Agreement with the dense computation, relative to the largest entry (rounding only).
TOL = 1e-12


def _keys(n: int) -> list[DataKey]:
    return [DataKey(f"2000-01-{1 + i % 28:02d}", f"WOPR:P{i}") for i in range(n)]


def _ensemble(seed: int, n_state: int, n_data: int, n_ens: int) -> tuple[Any, ...]:
    rng = np.random.default_rng(seed)
    state = rng.standard_normal((n_state, n_ens))
    # Correlated data, so that the taper keeps some entries and drops others.
    predicted = rng.standard_normal((n_data, n_state)) @ state / np.sqrt(n_state)
    predicted += 0.5 * rng.standard_normal((n_data, n_ens))
    gain = rng.standard_normal((n_ens, n_data))
    innovations = rng.standard_normal((n_data, n_ens))
    return state, predicted, gain, innovations


def _mask(keys: Sequence[DataKey]) -> NDArray[np.bool_]:
    """A parameter mask on the 'poro' rows (23 x n_data): every third datum off."""
    mask = np.ones((23, len(keys)), dtype=bool)
    mask[:, ::3] = False
    return mask


def _prepared(options: dict[str, Any], restricted: bool, seed: int = 3) -> tuple[Any, ...]:
    state, predicted, gain, innovations = _ensemble(seed, sum(SIZES), N_DATA, N_ENS)
    loc = PriorAdaptiveLocalization(SIZES, variable_names=NAMES, **options)
    if restricted:
        loc.restrict(slice(37, 60), _mask)
    rng = np.random.default_rng(11)
    loc.prepare(state, predicted, _keys(N_DATA), rng)
    return loc, rng, state, gain, innovations


def _bits(a: NDArray[Any]) -> bytes:
    return np.ascontiguousarray(a).tobytes()


def _close(found: NDArray[Any], expected: NDArray[Any]) -> None:
    assert found.shape == expected.shape
    scale = max(float(np.abs(expected).max()), 1e-300)
    np.testing.assert_allclose(found, expected, rtol=0, atol=TOL * scale)


VARIANTS = [
    {"taper": "hard"},
    {"taper": "soft"},
    {"taper": "sigmoid"},
    {"taper": "hard", "threshold": "fixed"},
    {"taper": "soft", "threshold": "fixed", "cutoff": 0.2},
    {"taper": "hard", "unlocalized": ["nn"]},
    {"taper": "soft", "threshold_scale": 0.5, "unlocalized": ["aq"]},
]


@pytest.mark.parametrize("variant", VARIANTS, ids=lambda v: "-".join(map(str, v.values())))
@pytest.mark.parametrize("storage", ["auto", "packed", "recompute", "dense"])
@pytest.mark.parametrize("block_rows", [0, 1, 4, 7, 4096])
@pytest.mark.parametrize("restricted", [False, True])
def test_every_storage_and_block_size_agrees_with_the_dense_results(
    variant: dict[str, Any], storage: str, block_rows: int, restricted: bool
) -> None:
    ref, ref_rng, state, gain, innov = _prepared(
        {**variant, "taper_storage": "dense", "block_rows": 0}, restricted
    )
    loc, rng, *_ = _prepared(
        {**variant, "taper_storage": storage, "block_rows": block_rows}, restricted
    )
    assert ref.taper_matrix is not None
    _close(loc.dense_taper(), ref.taper_matrix)
    # The same draws from the localization's stream.
    assert rng.bit_generator.state == ref_rng.bit_generator.state
    anomalies = state - state.mean(axis=-1, keepdims=True)
    found = loc.localized_step(anomalies, gain, innov, None)
    _close(found, ref.localized_step(anomalies, gain, innov, None))
    gains = [gain * (1 + 0.1 * j) for j in range(N_ENS)]
    members = loc.localized_member_steps(anomalies, iter(gains), innov, None)
    _close(members, ref.localized_member_steps(anomalies, iter(gains), innov, None))
    # Checkpointed and restored in the same mode: the same steps, bit for bit.
    restored = PriorAdaptiveLocalization(
        SIZES, variable_names=NAMES,
        **{**variant, "taper_storage": storage, "block_rows": block_rows},
    )  # fmt: skip
    restored.restore_state(loc.export_state())
    assert _bits(restored.localized_step(anomalies, gain, innov, None)) == _bits(found)
    again = restored.localized_member_steps(anomalies, iter(gains), innov, None)
    assert _bits(again) == _bits(members)


def test_hopeless_columns_and_constant_rows_agree_with_the_dense_taper() -> None:
    """A large variable whose universal cutoff reaches 1 for some data (zeroed columns), a
    one-row variable, a constant state row and a constant datum (the 1e-6 std floor)."""
    sizes = [400, 1, 30]
    rng = np.random.default_rng(13)
    state = rng.standard_normal((431, 12))
    state[5] = 2.0  # constant
    predicted = rng.standard_normal((9, 431)) @ state / 20 + rng.standard_normal((9, 12))
    predicted[4] = -1.0  # constant
    anomalies = state - state.mean(axis=-1, keepdims=True)
    gain, innov = rng.standard_normal((12, 9)), rng.standard_normal((9, 12))
    variants: list[dict[str, Any]] = [
        {"taper": "hard"},
        {"taper": "soft"},
        {"taper": "sigmoid"},
        {"taper": "soft", "threshold": "fixed"},
    ]
    for options in variants:
        ref = PriorAdaptiveLocalization(sizes, **options, taper_storage="dense", block_rows=0)
        ref.prepare(state, predicted, _keys(9), np.random.default_rng(1))
        assert ref.taper_matrix is not None
        for storage in ("packed", "recompute"):
            for block_rows in (0, 3, 1000):
                loc = PriorAdaptiveLocalization(
                    sizes, **options, taper_storage=storage, block_rows=block_rows
                )
                loc.prepare(state, predicted, _keys(9), np.random.default_rng(1))
                _close(loc.dense_taper(), ref.taper_matrix)
                _close(
                    loc.localized_step(anomalies, gain, innov, None),
                    ref.localized_step(anomalies, gain, innov, None),
                )
    hard = PriorAdaptiveLocalization(sizes, taper_storage="dense", block_rows=0)
    hard.prepare(state, predicted, _keys(9), np.random.default_rng(1))
    assert hard.taper_matrix is not None
    assert np.any(np.all(hard.taper_matrix[:400] == 0.0, axis=0))  # a hopeless column


@pytest.mark.parametrize("storage", ["packed", "recompute"])
@pytest.mark.parametrize("taper", ["hard", "soft"])
def test_a_checkpoint_through_npz_resumes_bit_for_bit(
    storage: str, taper: str, tmp_path: Path
) -> None:
    loc, _, state, gain, innov = _prepared(
        {"taper": taper, "taper_storage": storage, "block_rows": 5}, True
    )
    np.savez(tmp_path / "prepared.npz", **loc.export_state())
    with np.load(tmp_path / "prepared.npz") as stored:
        arrays = {k: stored[k] for k in stored.files}
    options: dict[str, Any] = {"taper": taper, "taper_storage": storage, "block_rows": 5}
    restored = PriorAdaptiveLocalization(SIZES, variable_names=NAMES, **options)
    restored.restore_state(arrays)
    anomalies = state - state.mean(axis=-1, keepdims=True)
    assert _bits(restored.localized_step(anomalies, gain, innov, None)) == _bits(
        loc.localized_step(anomalies, gain, innov, None)
    )


class _Sink:
    """Diagnostics that want the taper but refuse anything over 1 kB, as max_mb does."""

    def __init__(self) -> None:
        self.calls: list[int | None] = []

    def add(self, role: str, name: str, item: object, *, nbytes: int | None = None) -> None:
        self.calls.append(nbytes)
        if nbytes is not None and nbytes > 1000:
            return
        if callable(item):
            item()

    def wants(self, role: str, name: str) -> bool:
        return True


def test_the_taper_diagnostic_states_its_size_before_a_packed_taper_is_built() -> None:
    """A diagnostic over its size limit must not build the whole taper (42 GB for Ula)."""
    loc, _, state, gain, innov = _prepared({"taper_storage": "packed", "block_rows": 4}, False)
    sink = _Sink()
    loc.attach_diagnostics(sink)
    built: list[tuple[int, int]] = []
    rows = loc._taper_rows

    def spy(start: int, stop: int) -> NDArray[np.float64]:
        built.append((start, stop))
        found: NDArray[np.float64] = rows(start, stop)
        return found

    loc._taper_rows = spy
    anomalies = state - state.mean(axis=-1, keepdims=True)
    loc.localized_step(anomalies, gain, innov, None)
    assert sink.calls == [sum(SIZES) * N_DATA * 8]
    assert (0, sum(SIZES)) not in built  # only blocks of 4 rows


@pytest.mark.parametrize("variant", VARIANTS[:3], ids=lambda v: v["taper"])
def test_a_packed_taper_exports_in_the_dense_ones_format(variant: dict[str, Any]) -> None:
    """The checkpoint format does not change: packed in memory is pack_taper's encoding
    (bit for bit for a hard taper, whose entries agree; doubles within rounding else)."""
    ref, *_ = _prepared({**variant, "taper_storage": "dense", "block_rows": 0}, True)
    loc, *_ = _prepared({**variant, "taper_storage": "packed", "block_rows": 5}, True)
    expected = pack_taper(ref.taper_matrix)
    found = loc.export_state()
    assert sorted(found) == sorted(expected)
    for key, value in expected.items():
        assert found[key].dtype == value.dtype, key
        if key == "taper_rows":
            _close(found[key], value)
        else:
            assert _bits(found[key]) == _bits(value), key
    assert _bits(found["taper_shape"]) == _bits(pack_taper(loc.dense_taper())["taper_shape"])


def test_a_dense_checkpoint_resumes_in_the_new_default_with_the_same_steps() -> None:
    """A run checkpointed before the option existed (dense, packed on disk) resumes exactly."""
    ref, _, state, gain, innov = _prepared({"taper_storage": "dense", "block_rows": 0}, False)
    resumed = PriorAdaptiveLocalization(SIZES, variable_names=NAMES)  # the defaults
    resumed.restore_state(ref.export_state())
    assert _bits(resumed.dense_taper()) == _bits(ref.taper_matrix)  # the same taper, exactly
    anomalies = state - state.mean(axis=-1, keepdims=True)
    _close(
        resumed.localized_step(anomalies, gain, innov, None),
        ref.localized_step(anomalies, gain, innov, None),
    )


def test_the_universal_noise_computed_in_column_chunks_is_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ref, *_ = _prepared({"taper_storage": "dense", "block_rows": 0}, False)
    whole, *_ = _prepared({"taper_storage": "recompute", "block_rows": 3}, False)
    monkeypatch.setattr(module, "_CORRELATION_ELEMENTS", 7)  # one or two columns at a time
    loc, *_ = _prepared({"taper_storage": "recompute", "block_rows": 3}, False)
    assert _bits(loc.dense_taper()) == _bits(whole.dense_taper())  # the medians are exact
    _close(loc.dense_taper(), ref.taper_matrix)


@pytest.mark.parametrize("storage", ["packed", "recompute"])
def test_larger_blocks_through_blas_give_the_same_step(storage: str) -> None:
    """Sizes where BLAS takes its blocked kernels (and an uneven last block)."""
    sizes = [1500, 1500, 3]
    state, predicted, gain, innov = _ensemble(7, sum(sizes), 400, 60)
    anomalies = state - state.mean(axis=-1, keepdims=True)
    steps = []
    options: list[dict[str, Any]] = [
        {"taper_storage": "dense", "block_rows": 0},
        {"taper_storage": storage, "block_rows": 256},
        {"taper_storage": storage, "block_rows": 1000},
    ]
    for option in options:
        loc = PriorAdaptiveLocalization(sizes, **option)
        loc.prepare(state, predicted, _keys(400), np.random.default_rng(2))
        steps.append(loc.localized_step(anomalies, gain, innov, None))
    _close(steps[1], steps[0])
    _close(steps[2], steps[0])


def test_blocks_keep_the_peak_memory_well_below_one_dense_taper(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The point of the options: no (n_state, n_data) array is formed."""
    n_state, n_data, n_ens = 20_000, 500, 20
    state, predicted, gain, innov = _ensemble(9, n_state, n_data, n_ens)
    anomalies = state - state.mean(axis=-1, keepdims=True)
    dense_bytes = n_state * n_data * 8  # 80 MB
    loc = PriorAdaptiveLocalization([n_state], taper_storage="auto", block_rows=1000)
    # The universal threshold's correlations a few columns at a time (the default is sized
    # for Ula's 200,713-row variables: 134 MB).
    monkeypatch.setattr(module, "_CORRELATION_ELEMENTS", 2**20)
    tracemalloc.start()
    try:
        loc.prepare(state, predicted, _keys(n_data), np.random.default_rng(0))
        loc.localized_step(anomalies, gain, innov, None)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    # The dense path holds three such arrays at once (the correlations, their shuffled
    # copy and the taper; then the taper, the gain and their product).
    assert peak < 0.5 * dense_bytes, f"peak {peak / 1e6:.1f} MB"


def test_the_local_analysis_selects_the_same_groups_from_a_packed_taper() -> None:
    ref, *_ = _prepared({"taper_storage": "dense", "block_rows": 0}, True)
    loc, *_ = _prepared({"taper_storage": "packed", "block_rows": 4}, True)
    n_state = sum(SIZES)
    dense = LocalAnalysisUpdate(localization=ref).groups(n_state, N_DATA)
    packed = LocalAnalysisUpdate(localization=loc).groups(n_state, N_DATA)
    assert len(dense) == len(packed)
    for (rows_a, data_a), (rows_b, data_b) in zip(dense, packed, strict=True):
        np.testing.assert_array_equal(rows_a, rows_b)
        np.testing.assert_array_equal(data_a, data_b)


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"taper_storage": "sparse"}, "taper_storage must be"),
        ({"block_rows": -1}, "block_rows must be"),
        ({"block_rows": True}, "block_rows must be"),
        ({"block_rows": 2.5}, "block_rows must be"),
    ],
)
def test_invalid_storage_options_are_refused(options: dict[str, Any], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        PriorAdaptiveLocalization([4], **options)


def test_auto_packs_a_hard_taper_and_recomputes_a_smooth_one() -> None:
    assert PriorAdaptiveLocalization([4]).storage == "packed"
    assert PriorAdaptiveLocalization([4], taper="soft").storage == "recompute"
    assert PriorAdaptiveLocalization([4], taper="sigmoid").storage == "recompute"
    assert PriorAdaptiveLocalization([4], taper_storage="dense").storage == "dense"
