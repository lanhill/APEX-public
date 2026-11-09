"""Build this example's data: the true model, its observations and the forecast deck.

The deck (``deck.mako``) is a small box with one water injector and two producers, three
years of history reported quarterly. This draws the true log-permeability, runs it once
with OPM Flow through APEX's simulator, and writes:

- ``truth.npz``: the true model (``permx``, ln mD; 800 cells);
- ``observations.csv``: the wells' rates at the 12 report times, with a standard deviation
  of 10 % of the rate, at least 8 (a rate of 0 is observed as 0, with 10²; a small
  positive rate may be observed slightly negative, as the noise allows);
- ``deck_forecast.mako``: ``deck.mako``'s history unchanged, then two years of forecast
  under the producers' bottom-hole pressures ``bhp`` of a control scenario
  (``predict_bhp.toml``).

::

    uv run python examples/prediction_controls/prepare.py

The files it writes are committed, so this only needs rerunning to refresh them.
"""

import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
#: The grid, and the true model's mean, standard deviation and range (cells) of ln PERMX.
GRID, TRUTH_MEAN, TRUTH_STD, TRUTH_RANGE = (20, 20, 2), 5.0, 1.0, 8.0
#: Seeds of the true model and of the observation noise.
TRUTH_SEED, NOISE_SEED = 7, 2061
#: A rate's relative error, the floor of its variance (8²) and that of a rate of 0 (10²).
RATE_RELATIVE, RATE_VARIANCE, ZERO_VARIANCE = 0.10, 64.0, 100.0
WELLS = {"WOPR": ("PRO1", "PRO2"), "WWPR": ("PRO1", "PRO2"), "WWIR": ("INJ1",)}

#: The forecast after the history: each producer at its scenario's bottom-hole pressure.
FORECAST = """-- The forecast (written by prepare.py): the producers' bottom-hole pressures (bar)
-- of the control scenario, bhp = [PRO1, PRO2] (the prediction's [[controls.variables]]),
-- from 2025-01-01, reported quarterly for two years.
WCONPROD
 'PRO1' 'OPEN' 'BHP' 5* ${bhp[0]} /
 'PRO2' 'OPEN' 'BHP' 5* ${bhp[1]} /
/
"""
#: The forecast's report times: quarterly to 2027-01-01.
FORECAST_DATES = [("APR", 2025), ("JUL", 2025), ("OCT", 2025), ("JAN", 2026),
                  ("APR", 2026), ("JUL", 2026), ("OCT", 2026), ("JAN", 2027)]  # fmt: skip


def write_truth(out: Path) -> dict[str, np.ndarray]:
    """Draw the true ln PERMX with APEX's ``gaussian_field`` sampler (``truth.npz``)."""
    from apex.algorithms.priors import GaussianFieldPrior
    from apex.domain.state import StateVariable

    field = GaussianFieldPrior(grid=list(GRID), mean=TRUTH_MEAN, variance=TRUTH_STD**2,
                               major_range=TRUTH_RANGE, variogram="gaussian")  # fmt: skip
    size = int(np.prod(GRID))
    permx = field.generate(
        StateVariable(name="permx", size=size), 2, np.random.default_rng(TRUTH_SEED)
    )
    truth = {"permx": permx[:, 0]}  # the first of two draws (the sampler needs two)
    np.savez(out / "truth.npz", **truth)
    return truth


def write_forecast_deck(out: Path) -> None:
    """Write ``deck_forecast.mako``: the history's deck, then the forecast before its END."""
    deck = (out / "deck.mako").read_text()
    end = deck.rindex("\nEND")
    schedule = "".join(f"DATES\n 1 '{m}' {y} /\n/\n" for m, y in FORECAST_DATES)
    (out / "deck_forecast.mako").write_text(deck[:end] + "\n\n" + FORECAST + schedule + deck[end:])


def simulate_truth(out: Path, truth: dict[str, np.ndarray], work: Path) -> Path:
    """Run the true model with OPM Flow in ``work``; return its case path."""
    from apex.core.simulation import MemberInput
    from apex.simulation.opm_flow import OPMFlowSimulator

    simulator = OPMFlowSimulator(deck_template_file=str(out / "deck.mako"), case_dir=str(out))
    return simulator.simulate(MemberInput(-1, truth, work))


def write_observations(out: Path, case: Path) -> None:
    """Write the rate observations, drawn around the truth's rates."""
    from resdata.summary import Summary

    from apex.domain.observations import DataKey
    from apex.simulation.deck import read_summary

    rng = np.random.default_rng(NOISE_SEED)
    dates = [d.isoformat()[:10] for d in Summary(str(case)).report_dates]
    keys = [DataKey(t, f"{v}:{w}") for t in dates for v, wells in WELLS.items() for w in wells]
    rates = read_summary(case, keys)
    lines = ["time,datatype,value,variance"]
    for key, rate in zip(keys, rates, strict=True):
        variance = max((RATE_RELATIVE * rate) ** 2, RATE_VARIANCE) if rate > 0 else ZERO_VARIANCE
        observed = float(rate + np.sqrt(variance) * rng.standard_normal()) if rate > 0 else 0.0
        lines.append(f"{key.time},{key.datatype},{observed!r},{variance}")
    (out / "observations.csv").write_text("\n".join(lines) + "\n")


def main(out: Path = HERE) -> None:
    """Draw and run the truth, and write the observations and the forecast deck."""
    truth = write_truth(out)
    write_forecast_deck(out)
    with tempfile.TemporaryDirectory() as work:
        write_observations(out, simulate_truth(out, truth, Path(work)))
    print(f"wrote the truth, the observations and the forecast deck to {out}")


if __name__ == "__main__":
    main()
