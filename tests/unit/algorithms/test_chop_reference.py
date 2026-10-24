"""CHOP's kernels against the author's MATLAB code.

The reference ``tests/data/reference/chop/chop_step.npz`` is one trial of the author's
``RLM_MAC_CHOP.m`` (5Spots) and its ``update_hyper.m`` / ``update_hyper_no_loc.m``, on a
small linear case, from ``tools/chop_reference/generate.py`` (its README entry says which
backend: a line-by-line transcription of the MATLAB, as this host's MATLAB licence was
refused). APEX's pieces that ``chop`` uses (RLM-MAC's gain, the per-member tapers) are run
on the same inputs with the code's two documented differences given explicitly:

- the code's ``getCorrMtx`` returns ``(N - 1) / N`` times the sample correlation;
- the code's taper scale is ``2 (l + 1e-6)`` (``factor = 2`` and the ``+ 1e-6``).
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from apex.algorithms.data_assimilation.chop.components import ComponentContext, LengthScale
from apex.algorithms.data_assimilation.chop.taper import ChopTaper, HyperTaper, _normalised
from apex.algorithms.data_assimilation.data_error import DataErrorCovariance
from apex.algorithms.data_assimilation.gies import RLMMACUpdate
from apex.algorithms.data_assimilation.localization import correlation

REFERENCE = Path(__file__).parents[2] / "data" / "reference" / "chop" / "chop_step.npz"

# APEX scales the anomalies by 1/sqrt(N - 1) and the damping by the mean of the kept
# singular values squared; MATLAB uses unscaled anomalies and singular values. The two are
# equal algebraically (brief section 3.2); the differences are rounding, ~1e-14.
RTOL, ATOL = 1e-9, 1e-11


@pytest.fixture(scope="module")
def ref() -> dict[str, np.ndarray]:
    loaded = np.load(REFERENCE)
    return {k: loaded[k] for k in loaded.files}


@dataclass(frozen=True)
class Parts:
    """The reference's inputs in APEX's layout (members only, ensemble last)."""

    ne: int
    state: np.ndarray
    hyper: np.ndarray
    predicted: np.ndarray
    reference: np.ndarray
    new: np.ndarray
    new_reference: np.ndarray
    perturbed: np.ndarray
    damping: float
    energy: float
    identity: DataErrorCovariance


def _parts(ref: dict[str, np.ndarray]) -> Parts:
    nm = int(ref["in_nm"])
    ensemble, sim = ref["in_ensemble"], ref["in_simData"]
    ne = ensemble.shape[1] - 1
    return Parts(
        ne=ne, state=ensemble[:nm, :ne], hyper=ensemble[nm:, :ne], predicted=sim[:, :ne],
        reference=sim[:, -1], new=ref["in_simDataNew"][:, :ne],
        new_reference=ref["in_simDataNew"][:, -1], perturbed=ref["in_perturbedData"],
        damping=float(ref["in_lambda"]), energy=float(ref["in_tsvdData"]),
        identity=DataErrorCovariance.diagonal(np.ones(sim.shape[0])),
    )  # fmt: skip


def _context(n_state: int, n_data: int, ne: int) -> ComponentContext:
    return ComponentContext(
        n_state=n_state, n_data=n_data, block_sizes=(n_state,), datatypes=("y",) * n_data,
        n_ensemble=ne,
    )  # fmt: skip


def test_the_codes_correlations_are_the_shrunk_sample_correlations(
    ref: dict[str, np.ndarray],
) -> None:
    p = _parts(ref)

    np.testing.assert_allclose(
        ref["out_model_corr"], (p.ne - 1) / p.ne * correlation(p.state, p.predicted),
        rtol=1e-12, atol=1e-14,
    )  # fmt: skip


def test_the_model_update_is_the_codes(ref: dict[str, np.ndarray]) -> None:
    """RLM_MAC_CHOP.m lines 177-271: each member's taper on the shared RLM-MAC gain."""
    p = _parts(ref)
    n_data = p.predicted.shape[0]
    taper = ChopTaper([LengthScale(mode="data", correlation="current", factor=2.0)])
    taper.configure(_context(p.state.shape[0], n_data, p.ne), [slice(0, n_data)])
    taper.theta = p.hyper + 1e-6  # the code's l + 1e-6
    taper._current = (_normalised(p.state) * (p.ne - 1) / p.ne, _normalised(p.predicted))
    strategy = RLMMACUpdate(energy=p.energy, localization=taper)

    step = strategy.step(
        p.state, p.predicted, p.perturbed, p.identity, damping=p.damping,
        reference_prediction=p.reference,
    )  # fmt: skip

    np.testing.assert_allclose(p.state + step, ref["out_model"], rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("localized", [False, True])
def test_the_hyper_parameter_update_is_the_codes(
    ref: dict[str, np.ndarray], localized: bool
) -> None:
    """update_hyper.m (localized) and update_hyper_no_loc.m: an RLM-MAC step on the length
    scales from the new forecast, the length scales floored at 1e-6."""
    p = _parts(ref)
    taper = None
    if localized:
        taper = HyperTaper([LengthScale(mode="data", factor=2.0)])
        taper.configure(_context(1, p.new.shape[0], p.ne), [slice(0, p.hyper.shape[0])])
        taper.theta = p.hyper + 1e-6
        taper.correlation = correlation(p.hyper, p.new) * (p.ne - 1) / p.ne
    strategy = RLMMACUpdate(energy=p.energy, localization=taper)

    step = strategy.step(
        p.hyper, p.new, p.perturbed, p.identity, damping=p.damping,
        reference_prediction=p.new_reference,
    )  # fmt: skip

    expected = ref["out_hyper_localized" if localized else "out_hyper_unlocalized"]
    assert (expected == 1e-6).any()  # the case makes the floor bind
    np.testing.assert_allclose(np.maximum(p.hyper + step, 1e-6), expected, rtol=RTOL, atol=ATOL)


def test_the_case_truncates_and_the_hyper_correlations_are_the_codes(
    ref: dict[str, np.ndarray],
) -> None:
    """The reference keeps fewer directions than data (the strict > rule matters), and its
    hyper-parameter correlations are corr(l^i, g(m^{i+1})) shrunk by (N - 1) / N."""
    p = _parts(ref)

    assert int(ref["out_model_rank"]) < min(p.new.shape[0], p.ne - 1)
    np.testing.assert_allclose(
        ref["out_hyper_corr"], (p.ne - 1) / p.ne * correlation(p.hyper, p.new),
        rtol=1e-12, atol=1e-14,
    )  # fmt: skip
