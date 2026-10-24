"""C-GIES against a NumPy transcription of the user's MATLAB ``GIES_constraint_ensLoc.m``.

The independent reference is ``tools/gmac_reference/generate_cgies.py``: a line-by-line
transcription of the script's update (its ``(M_j + lambda tr(M_j)/N I)^-1``, trace
weights and per-member constraint metrics) with the scripts' channel and barrier, and
fixed histogram bins. Stored outputs: ``tests/data/reference/gmac/cgies_*``.
"""

from pathlib import Path

import numpy as np
import pytest

from apex.algorithms.data_assimilation import (
    AdaptiveLocalization,
    DataErrorCovariance,
    GMACUpdate,
)

REFERENCE = Path(__file__).resolve().parents[2] / "data" / "reference" / "gmac"


@pytest.mark.parametrize("case", ["cgies_in", "cgies_eq", "cgies_in_eq", "cgies_in_eq_localized"])
def test_cgies_matches_the_matlab_block(
    case: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``_localized``: the script's per-source tapers (``autoAdaLoc``'s, random there) are
    given, in its order (data, EQ, IN), and injected into ``adaptive_per_update``."""
    ref = np.load(REFERENCE / f"{case}.npz")
    ensemble, sim_data = ref["in_ensemble"], ref["in_sim_data"]
    n, size = ensemble.shape[1] - 1, ensemble.shape[0]
    use_eq, use_in = (bool(v) for v in ref["in_use"])
    w_eq, w_in = (float(v) for v in ref["in_weights"])
    np.savez(tmp_path / "truth.npz", m=ref["in_truth"])
    constraints = []
    if use_eq:
        constraints.append({
            "function": {"name": "histogram", "options": {
                "variable": "m", "range": list(ref["in_hist_range"]),
                "bins": int(ref["in_bins"]), "reference_file": str(tmp_path / "truth.npz")}},
            "metric": {"name": "log_channel",
                       "options": {"b": 0.1, "epsilon": 0.001, "curvature": "paper"}},
            "weight": {"rule": "trace", "w": w_eq},
        })  # fmt: skip
    if use_in:
        constraints.append({
            "function": {"name": "box", "options": {"bounds": {"m": list(ref["in_bounds"])}}},
            "metric": {"name": "log_barrier", "options": {"a": 0.1}},
            "weight": {"rule": "trace", "w": w_in},
        })  # fmt: skip
    localization = None
    if ref["in_tapers"].size:
        tapers = list(ref["in_tapers"])
        monkeypatch.setattr(AdaptiveLocalization, "_taper", lambda *args: tapers.pop(0))
        localization = AdaptiveLocalization([size])
    gmac = GMACUpdate(
        code="100", form="direct", state_variables=[{"name": "m", "size": size}],
        constraints=constraints, localization=localization,
    )  # fmt: skip

    step = gmac.step(
        np.ascontiguousarray(ensemble[:, :n]),
        np.ascontiguousarray(sim_data[:, :n]),
        ref["in_perturbed_data"],
        DataErrorCovariance.diagonal(np.ones(sim_data.shape[0])),  # the inputs are whitened
        damping=float(ref["in_lambda"]),
        reference_prediction=sim_data[:, -1].copy(),
        rng=np.random.default_rng(0),
    )

    # the same operations up to pinv vs inv and the sources' summation order: measured
    # 1.2e-15 to 3.0e-15 of the largest step entry (~1e3), so atol 1e-8 is ~1e-11 relative.
    np.testing.assert_allclose(
        ensemble[:, :n] + step, ref["out_iterated_ensemble"], rtol=1e-10, atol=1e-8
    )
