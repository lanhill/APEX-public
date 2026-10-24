"""The wavelet transform's portable state and member-wise property.

Independent reference: the transform fitted here, applied to the whole ensemble. A
transform rebuilt from its options and loaded with the fitted state must give the same
coefficients bit for bit, and so must compressing one member (column) at a time.
"""

import numpy as np
import pytest

from apex.algorithms.data_compression import WaveletCompression
from apex.core.simulation import PortableTransform
from apex.domain.observations import DataKey, ObservationSet


def _observations(
    grid: tuple[int, ...], rng: np.random.Generator, n: int | None = None
) -> ObservationSet:
    n = int(np.prod(grid)) if n is None else n
    field = 10 * np.cumsum(rng.standard_normal(n)) + rng.standard_normal(n)
    keys = [DataKey("t0", "y")] + [DataKey("t1", "avo", k) for k in range(n)]
    keys += [DataKey("t2", "y")]
    values = np.concatenate([[1.0], field, [2.0]])
    return ObservationSet(keys, values, np.full(len(keys), 0.01))


@pytest.mark.parametrize(
    ("grid", "wavelet", "level", "order", "thresholding", "masked"),
    [((16, 16), "haar", 2, "F", "hard", False), ((13, 7, 5), "db2", 1, "C", "hard", False),
     ((24, 16, 32), "sym4", 2, "F", "hard", False), ((13, 7, 5), "sym4", 1, "F", "soft", True),
     ((11, 6), "bior2.2", 2, "C", "soft", True)],
)  # fmt: skip
def test_a_loaded_state_and_one_member_at_a_time_give_the_ensembles_coefficients(
    grid: tuple[int, ...],
    wavelet: str,
    level: int,
    order: str,
    thresholding: str,
    masked: bool,
) -> None:
    """With soft thresholding and inactive cells (a mask, with a given inactive value). The
    codec remote members carry the state by is checked with the executor that uses it."""
    rng = np.random.default_rng(3)
    n = int(np.prod(grid))
    mask = rng.random(n) > 0.25 if masked else None
    options = {
        "datatype": "avo", "grid": grid, "wavelet": wavelet, "level": level, "order": order,
        "keep_approximation": True, "threshold_multiplier": 0.05,
        "thresholding": thresholding, "mask": mask,
        "inactive_value": -3.25 if masked else None,
    }  # fmt: skip
    fitted = WaveletCompression(**options)  # type: ignore[arg-type]
    observations = _observations(grid, rng, None if mask is None else int(mask.sum()))
    fitted.fit(observations)
    predicted = rng.standard_normal((observations.n_data, 9))
    whole = fitted.apply(predicted)

    loaded = WaveletCompression(**options)  # type: ignore[arg-type]
    assert isinstance(loaded, PortableTransform)
    loaded.load_state(fitted.state())
    assert loaded.apply(predicted).tobytes() == whole.tobytes()
    for j in range(predicted.shape[1]):
        column = loaded.apply(np.ascontiguousarray(predicted[:, [j]]))[:, 0]
        assert column.tobytes() == np.ascontiguousarray(whole[:, j]).tobytes(), j


@pytest.mark.parametrize("change", [{"order": "C"}, {"thresholding": "soft"}, {"level": 3}])
def test_a_state_for_other_options_is_refused(change: dict[str, object]) -> None:
    """The numerics review of phase 3: order and thresholding change every member's
    coefficients, so a state fitted for other options must not load."""
    rng = np.random.default_rng(4)
    options: dict[str, object] = {"keep_approximation": True, "threshold_multiplier": 0.05}
    fitted = WaveletCompression("avo", (16, 16), "haar", 2, **options)  # type: ignore[arg-type]
    fitted.fit(_observations((16, 16), rng))
    other = WaveletCompression("avo", (16, 16), "haar", **{"level": 2, **options, **change})  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="other options"):
        other.load_state(fitted.state())


@pytest.mark.parametrize("broken", ["active", "retained", "observed", "source"])
def test_a_state_whose_arrays_do_not_fit_together_is_refused(broken: str) -> None:
    rng = np.random.default_rng(5)
    fitted = WaveletCompression("avo", (16, 16), "haar", 2, keep_approximation=True,
                                threshold_multiplier=0.05)  # fmt: skip
    fitted.fit(_observations((16, 16), rng))
    state = fitted.state()
    if broken == "active":
        state["1.active"] = state["1.active"][:-5]
    elif broken == "retained":
        state["1.retained"] = state["1.retained"] + 10**7
    elif broken == "observed":
        state["1.observed"] = state["1.observed"][:-1]
    else:
        state["1.source"] = state["1.source"] + 1
    other = WaveletCompression("avo", (16, 16), "haar", 2, keep_approximation=True,
                               threshold_multiplier=0.05)  # fmt: skip
    with pytest.raises(ValueError, match="wavelet"):
        other.load_state(state)
