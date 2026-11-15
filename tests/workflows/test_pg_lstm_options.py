"""``pg_lstm``'s causal mask, saved inputs and report dates.

The mask is checked against finite differences through the model's own forward pass
(``well_residuals``): a parameter moves a datum exactly where the mask says, with every
residual kept (``shut_wells = "ignore"``) and a superset of the moved data with shut wells
zeroed. Inputs are planted block states, so nothing is simulated.
"""

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from apex.core.errors import SimulationError
from apex.core.simulation import MaskingModel
from apex.domain.observations import DataKey, ObservationSet
from apex.simulation.model_error import INPUT_PREFIX, RESIDUAL_PREFIX
from apex.workflows.model_error import PGLSTMModel

# Report steps 1..4: TSTEP 1 10 (steps 1, 2), then two DATES (steps 3, 4); P1 shut at step 2.
SCHEDULE = """\
COMPDAT
'P1' 1 1 1 1 OPEN /
'P1' 2 1 1 1 OPEN /
'I1' 3 1 1 1 OPEN /
/
WCONHIST
'P1' 'OPEN' 'LRAT' 100 /
/
WCONINJH
'I1' 'WATER' 'OPEN' 100 /
/
TSTEP
1 /
WCONHIST
'P1' 'SHUT' 'LRAT' 0 /
/
TSTEP
10 /
WCONHIST
'P1' 'OPEN' 'LRAT' 100 /
/
DATES
1 'FEB' 2000 /
/
DATES
1 'MAR' 2000 /
/
"""
DATES = ("2000-01-02", "2000-01-12", "2000-02-01", "2000-03-01")
STEPS = [1, 2, 3, 4]
OUTPUTS = {"P1": ("WOPR", "WBHP", "WWCT"), "I1": ("WBHP",)}


def _model(tmp_path: Path, **options: object) -> PGLSTMModel:
    path = tmp_path / "Hist"
    path.write_text(SCHEDULE)
    model = PGLSTMModel(schedule_file=path, report_steps=STEPS, producers=["P1"],
                        injectors=["I1"], hidden_size=3, **options)  # type: ignore[arg-type]  # fmt: skip
    keys = _keys()
    model.fit(ObservationSet(keys, np.linspace(1.0, 2.0, len(keys)), np.ones(len(keys))))
    return model


def _keys(dates: Sequence[str] = DATES) -> list[DataKey]:
    return [DataKey(d, f"{o}:{w}") for d in dates for w, outs in OUTPUTS.items() for o in outs]


class _States:
    """Planted block states at three cells (i = 1, 2, 3), as ``CaseFiles`` reads them."""

    label = "planted case"

    def __init__(self, seed: int = 0, dates: Sequence[str] = DATES) -> None:
        rng = np.random.default_rng(seed)
        self.dates = list(dates)
        self.statics = {n: rng.uniform(1, 2, 3) for n in
                        ("DX", "DY", "DZ", "PERMX", "PERMY", "PERMZ", "PORO")}  # fmt: skip
        self.dynamics = [{n: rng.uniform(0.1, 1, 3) for n in
                          ("PRESSURE", "SOIL", "SWAT", "OIL_DEN", "WAT_DEN")}
                         for _ in STEPS]  # fmt: skip
        self.bhp = rng.uniform(0.5, 1.5, len(STEPS))  # small inputs: no saturated gate

    def active(self, block: tuple[int, int, int]) -> int:
        return block[0] - 1

    def static(self, name: str) -> NDArray[np.float64]:
        return self.statics[name]

    def restart(self, step: int) -> dict[str, NDArray[np.float64]]:
        return self.dynamics[step - 1]

    def wbhp(self, well: str, step: int) -> float:
        return float(self.bhp[step - 1])

    def report_dates(self, steps: Sequence[int]) -> list[str]:
        return [self.dates[s - 1] for s in steps]


@pytest.mark.parametrize("shut_wells", ["ignore", "zero"])
def test_the_mask_is_where_each_parameter_moves_the_data(tmp_path: Path, shut_wells: str) -> None:
    """Finite differences through the forward pass: nothing moves outside the mask.

    The mask is the structure, a superset of the moved data where the sensitivity is zero
    for other reasons (exact in the sense: false only where it is
    zero). Every output layer's pairs are moved exactly. Of the LSTM's, at the first report
    time ``h_0 = c_0 = 0``, so the recurrent weights and the forget gate move nothing; at
    later times all but a rounding coincidence move. With shut wells zeroed a shut step's
    rates do not move at all, while the mask keeps them.
    """
    model = _model(tmp_path, report_dates=DATES, causal_mask=True, shut_wells=shut_wells)
    assert isinstance(model, MaskingModel)
    assert model.masks_parameters
    keys = _keys()
    mask = model.parameter_mask(keys)
    assert mask.shape == (model.n_params, len(keys))
    states = _States()
    # Small weights and a large bump: no gate saturates, so a float32 change is visible.
    theta = np.random.default_rng(3).normal(0.0, 0.1, model.n_params)
    base = model.state_residuals(states, theta, keys)
    moved = np.zeros_like(mask)
    for p in range(model.n_params):
        bumped = theta.copy()
        bumped[p] += 0.5
        moved[p] = model.state_residuals(states, bumped, keys) != base
    assert not np.any(moved & ~mask)
    lstm = np.zeros(model.n_params, dtype=bool)
    for network, part in zip(model.networks, model._slices, strict=True):
        h = network.hidden_size
        lstm[part.start : part.start + 4 * h * (network.input_size + h) + 8 * h] = True
    shut = np.array([k.time == DATES[1] and k.datatype in ("WOPR:P1", "WWCT:P1") for k in keys])
    later = np.array([k.time != DATES[0] for k in keys]) & ~(shut & (shut_wells == "zero"))
    np.testing.assert_array_equal(mask[~lstm], moved[~lstm] | (mask[~lstm] & ~later & shut))
    unmoved_later = (mask & ~moved)[np.ix_(lstm, later)]
    assert unmoved_later.sum() <= 0.002 * mask[np.ix_(lstm, later)].sum()
    if shut_wells == "zero":
        assert mask[:, shut].any()
        assert not moved[:, shut].any()


def test_the_mask_takes_any_subset_and_names_a_key_off_the_dates(tmp_path: Path) -> None:
    model = _model(tmp_path, report_dates=DATES, causal_mask=True)
    keys = [*_keys(), DataKey(DATES[0], "FOPR"), DataKey(DATES[0], "WOPR:P9")]
    full = model.parameter_mask(keys)
    assert not full[:, -2:].any()  # a field datum and another well's: no parameter's
    chosen = list(range(len(keys) - 1, -1, -2))
    np.testing.assert_array_equal(model.parameter_mask([keys[i] for i in chosen]),
                                  full[:, chosen])  # fmt: skip
    with pytest.raises(ValueError, match="not at one of the report dates"):
        model.parameter_mask([DataKey("2000-01-05", "WOPR:P1")])


def test_the_options_default_to_the_published_model(tmp_path: Path) -> None:
    """Without them the model declares no mask and answers only the residual keys."""
    model = _model(tmp_path)
    assert not model.masks_parameters
    keys = _keys()
    assert model.residual_keys(keys) == [DataKey(k.time, RESIDUAL_PREFIX + k.datatype)
                                         for k in keys]  # fmt: skip
    states, theta = _States(), np.random.default_rng(0).normal(0, 0.1, model.n_params)
    dated = _model(tmp_path, report_dates=DATES)
    np.testing.assert_array_equal(model.state_residuals(states, theta, keys),
                                  dated.state_residuals(states, theta, keys))  # fmt: skip


@pytest.mark.parametrize(
    ("options", "match"),
    [({"causal_mask": True}, "causal_mask needs report_dates"),
     ({"save_inputs": True}, "save_inputs needs report_dates"),
     ({"report_dates": DATES[:3]}, "4 increasing dates"),
     ({"report_dates": (DATES[1], DATES[0], *DATES[2:])}, "4 increasing dates"),
     ({"report_dates": ("2000-01-02", "x", "y", "z")}, "must be ISO dates")],
)  # fmt: skip
def test_bad_report_dates_are_refused(
    tmp_path: Path, options: dict[str, object], match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        _model(tmp_path, **options)


def test_the_saved_inputs_are_the_networks_inputs(tmp_path: Path) -> None:
    model = _model(tmp_path, report_dates=DATES, save_inputs=True)
    keys = _keys()
    answered = model.residual_keys(keys)
    inputs = [k for k in answered if k.datatype.startswith(RESIDUAL_PREFIX + INPUT_PREFIX)]
    sizes = [n.input_size for n in model.networks]
    assert len(inputs) == len(DATES) * sum(sizes)
    states = _States()
    theta = np.random.default_rng(0).normal(0, 0.1, model.n_params)
    values = model.state_residuals(states, theta, answered)
    features = dict(zip(["P1", "I1"], model.features(states), strict=True))
    for key, value in zip(answered, values, strict=True):
        if key in inputs:
            j, _, well = key.datatype.removeprefix(RESIDUAL_PREFIX + INPUT_PREFIX).partition(":")
            assert value == features[well][DATES.index(key.time), int(j)]
    # The residuals themselves are those of the model without the option.
    plain = _model(tmp_path, report_dates=DATES)
    n = len(keys)
    np.testing.assert_array_equal(values[:n], plain.state_residuals(states, theta, answered[:n]))


def test_a_case_reporting_other_dates_is_refused(tmp_path: Path) -> None:
    model = _model(tmp_path, report_dates=DATES, causal_mask=True)
    shifted = _States(dates=(*DATES[:3], "2000-03-02"))
    with pytest.raises(SimulationError, match="not report_dates"):
        model.state_residuals(shifted, np.zeros(model.n_params), _keys())
