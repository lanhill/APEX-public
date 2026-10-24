"""AVO traces against SimulatorWrap's ``flow_avo``, with 3dBox's settings.

The reference (``tests/data/reference/geophysics/avo_traces.npz``) is
``_calc_avo_props_active_cells`` at ``d29dc57`` on a 7 x 6 x 5 grid. APEX's ``nearest``
sampling reproduces it; the default ``interfaces`` sampling does not (it is continuous in
the model), and the difference is recorded. The reference's multi-angle defect (G7) is
pinned.
"""

from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from apex.algorithms.geophysics import (
    ApproxZoeppritzPP,
    Columns,
    ConstantOverburden,
    RickerWavelet,
    SeismicModel,
    TimeAxis,
)

pytestmark = pytest.mark.pet_parity

REFERENCE = Path(__file__).resolve().parents[1] / "data" / "reference" / "geophysics"
type Floats = NDArray[np.float64]


@pytest.fixture(scope="module")
def ref() -> dict[str, Floats]:
    path = REFERENCE / "avo_traces.npz"
    if not path.exists():
        pytest.fail(f"missing {path}; run tools/geophysics_reference/generate.py avo_traces")
    with np.load(path) as data:
        return {k: data[k] for k in data.files}


def _columns(ref: dict[str, Floats]) -> Columns:
    """The reference's (nz, ny, nx) arrays as columns, traces in (y, x) C order."""
    nz = ref["in_dz"].shape[0]

    def col(a: Floats) -> Floats:
        out: Floats = a.reshape(nz, -1).T
        return out

    return Columns.of(
        ref["in_top"].ravel(), col(ref["in_dz"]), col(ref["in_vp"]), col(ref["in_vs"]),
        col(ref["in_rho"]),
    )  # fmt: skip


def _model(sampling: str) -> SeismicModel:
    return SeismicModel(
        reflectivity=ApproxZoeppritzPP(),
        wavelet=RickerWavelet(45.0, 0.05),
        overburden=ConstantOverburden(2450.0, 849.0, 2300.0),
        angles_deg=(10.0, 20.0),
        time=TimeAxis(1.9224, 2.2150, 0.001),
        sampling=sampling,  # type: ignore[arg-type]
    )


def test_nearest_reproduces_flow_avo(ref: dict[str, Floats]) -> None:
    """Each angle as the reference computes it alone, in float64 throughout (PyLops'
    ``Convolve1D`` dtype only labels the operator); measured ≤ 1e-15 of the peak."""
    traces = _model("nearest").traces(_columns(ref))
    for i, angle in enumerate((10, 20)):
        expected = ref[f"out_traces_{angle}"].reshape(traces.shape[0], -1)
        scale = np.abs(expected).max()
        np.testing.assert_allclose(traces[:, i], expected, rtol=0, atol=1e-12 * scale)


def test_flow_avo_returns_the_last_angle_for_every_angle(ref: dict[str, Floats]) -> None:
    """G7: with two angles, the reference returns the 20° traces twice."""
    both = ref["out_traces_both"]
    np.testing.assert_array_equal(both[..., 0], ref["out_traces_20"])
    np.testing.assert_array_equal(both[..., 1], ref["out_traces_20"])
    assert not np.array_equal(ref["out_traces_10"], ref["out_traces_20"])


def test_interfaces_differ_from_the_staircase_by_a_bounded_amount(
    ref: dict[str, Floats],
) -> None:
    """The default sampling is not the reference's: about 13 % of the peak on this grid
    (measured), from the 0.5 ms staircase. Recorded so that a change shows."""
    columns = _columns(ref)
    exact = _model("interfaces").traces(columns)
    stair = _model("nearest").traces(columns)
    relative = np.abs(exact - stair).max() / np.abs(exact).max()
    assert 0.02 < relative < 0.25
