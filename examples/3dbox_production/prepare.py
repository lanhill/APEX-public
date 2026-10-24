"""Build this example's data from the 3dBox case: production data only.

The case is the APEX authors' own, set up for PET, and ships here under GPL-3.0-or-later:
its PVT tables (``pvt.txt``), well schedule (``Schdl.sch``) and true model (``truth.npz``,
the log-permeability and the porosity; ``poro.inc`` holds the porosity for decks run
without a porosity state). The same case with AVO data is ``examples/3dbox_production_avo``,
with the same deck and the same rates. The deck reports on the first of every month from
February 2022 to December 2032 and at the AVO example's two vintages (its ``DATES``). This
runs the truth once with OPM Flow through APEX's simulators, and writes from it:

- ``observations.csv``: the wells' rates at the 133 report times, with a standard deviation of
  10 % of the rate, at least 8 (a rate of 0 is observed as 0, with 10²);
- ``true_rates.csv``: the truth's noise-free rates (for ``synthetic_demo.py``'s figures);
- ``config_esmda.toml``, ``config_gies.toml``, ``config_esmda_noloc.toml`` and
  ``config_gies_noloc.toml``: ES-MDA and GIES, with and without localization.

::

    uv run python examples/3dbox_production/prepare.py

The files it writes are committed, so this only needs rerunning to refresh them.
"""

import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
#: Seed of the observation noise (the AVO example's, so the rates are the same).
SEED = 2061
#: The floor of a rate's variance (8², the PET set-up's), and that of a rate of 0 (10²).
RATE_VARIANCE, ZERO_VARIANCE = 64.0, 100.0
#: A rate's relative error (standard deviation over the rate).
RATE_RELATIVE = 0.10
#: The cap on ln PERMX in the deck (the PET deck's is 6, below the truth's 6.9).
PERMX_CAP = 8.0
#: The prior: members, seed, the Gaussian variogram's practical range (cells), ln PERMX's
#: and the porosity's mean and standard deviation, their correlation, and the bounds.
MEMBERS, PRIOR_SEED, PRIOR_RANGE = 100, 2026, 17.0
PERMX_PRIOR, PORO_PRIOR, RHO = (5.0, 1.0), (0.2, 0.05), 0.7
PERMX_BOUNDS, PORO_BOUNDS = (-3.0, 10.0), (0.01, 0.4)
WELLS = {"WOPR": ("PRO1", "PRO2", "PRO3"), "WWPR": ("PRO1", "PRO2", "PRO3"),
         "WWIR": ("INJ1", "INJ2", "INJ3")}  # fmt: skip


CONFIG = """# The 3dBox case, written by prepare.py: {title} on the wells' rates, on OPM Flow; the
# log-permeability and the porosity are uncertain.
[run]
name = "3dbox_production_{label}"
seed = 2026

[[state.variables]]
name = "permx"   # ln PERMX (mD); the deck caps it at permx_cap
size = 18000
grid = [60, 60, 5]
lower = -3.0     # bounds just outside the truth's -2.2 ... 6.9: members are clipped to them
upper = 10.0

[[state.variables]]
name = "poro"
size = 18000
grid = [60, 60, 5]
lower = 0.01
upper = 0.4

[prior]   # 100 members drawn by prepare.py: ln PERMX and a porosity correlated with it (0.7)
generator = {{ name = "file", options = {{ ensemble_file = "prior_ensemble.npz" }} }}

[observations]
file = "observations.csv"

[case]
kind = "synthetic"
truth_file = "truth.npz"

[[metrics]]
name = "normalized_data_mismatch"

[[metrics]]
name = "rmse"

# 2-D SSIM per layer against the truth, with a fixed value range per variable (about the
# truth's), so that the score is comparable across iterations and runs.
[[metrics]]
name = "ssim"
label = "ssim_permx"
options = {{ per_layer = true, win_size = 7, variables = ["permx"], data_range = 10.0 }}

[[metrics]]
name = "ssim"
label = "ssim_poro"
options = {{ per_layer = true, win_size = 7, variables = ["poro"], data_range = 0.4 }}

[simulator]
name = "opm_flow"

[simulator.options]
deck_template_file = "deck.mako"
timeout_s = 600
keep_files = false
template_values = {{ permx_cap = {permx_cap} }}   # ln PERMX cap in the deck (PET's is 6)

[execution]
name = "process"
options = {{ workers = 10 }}

{method}"""

LOCALIZATION = (
    "# Adaptive localization with the fixed threshold 3/sqrt(N) (0.3 at 100 members): without\n"
    "# it the ensemble collapses; the universal threshold keeps almost nothing here.\n"
    'localization = { name = "adaptive_prior", options = { threshold = "fixed" } }\n'
)
ESMDA = (
    '[method]\nname = "esmda"\noptions = { inflation = [4, 4, 4, 4] }\n'
    'strategy = { name = "approx", options = { energy = 0.98 } }\n'
)
GIES = '[method]\nname = "gies"\nstrategy = { name = "rlmmac", options = { energy = 0.98 } }\n'
GIES_STOP = (
    "# Two small improvements in a row, not one, end the run (a damped first step is small).\n"
    "schedule = { confirm_convergence = true }\nmax_iterations = 10\n"
)
NO_LOCALIZATION = "# No localization: for comparison (the ensemble collapses).\n"
#: The methods, each its own configuration: (file label, title, [method] section).
METHODS = (
    ("esmda", "ES-MDA (4 steps)", ESMDA + LOCALIZATION),
    ("gies", "GIES (RLM-MAC, 10 iterations)", GIES + LOCALIZATION + GIES_STOP),
    ("esmda_noloc", "ES-MDA (4 steps) without localization", ESMDA + NO_LOCALIZATION),
    ("gies_noloc", "GIES (RLM-MAC, 10 iterations) without localization",
     GIES + NO_LOCALIZATION + GIES_STOP),
)  # fmt: skip


def write_prior(out: Path) -> None:
    """Draw the prior ensemble (``prior_ensemble.npz``): ln PERMX and a correlated porosity.

    Both from APEX's ``gaussian_field`` sampler with a Gaussian variogram of practical range
    ``PRIOR_RANGE`` cells, as smooth as the truth (its correlation 0.996, 0.77 and 0.28 at
    1, 5 and 10 cells); the porosity is ``mean + std (rho z_K + sqrt(1 - rho^2) z_P)``, so
    ``rho`` is its correlation with ln PERMX (the truth's is 0.95). Clipped to the bounds.
    """
    from apex.algorithms.priors import GaussianFieldPrior
    from apex.domain.state import StateVariable

    field = GaussianFieldPrior(grid=[60, 60, 5], mean=0.0, variance=1.0,
                               major_range=PRIOR_RANGE, variogram="gaussian")  # fmt: skip
    rng = np.random.default_rng(PRIOR_SEED)
    z_k, z_p = (field.generate(StateVariable(name=n, size=18000), MEMBERS, rng) for n in "kp")
    permx = np.clip(PERMX_PRIOR[0] + PERMX_PRIOR[1] * z_k, *PERMX_BOUNDS)
    poro = PORO_PRIOR[0] + PORO_PRIOR[1] * (RHO * z_k + np.sqrt(1 - RHO**2) * z_p)
    poro = np.clip(poro, *PORO_BOUNDS)
    # float32: ample for a prior, and half the size of the committed file.
    np.savez_compressed(
        out / "prior_ensemble.npz", permx=permx.astype(np.float32), poro=poro.astype(np.float32)
    )


def write_configs(out: Path) -> None:
    """Write ``config_<method>.toml`` for each method."""
    for label, title, method in METHODS:
        text = CONFIG.format(title=title, label=label, permx_cap=PERMX_CAP, method=method)
        (out / f"config_{label}.toml").write_text(text)


def simulate_truth(out: Path, work: Path) -> Path:
    """Run the true model with OPM Flow in ``work``; return its case path."""
    from apex.core.simulation import MemberInput
    from apex.simulation.opm_flow import OPMFlowSimulator

    simulator = OPMFlowSimulator(
        deck_template_file=str(out / "deck.mako"), case_dir=str(out),
        template_values={"permx_cap": PERMX_CAP},
    )  # fmt: skip
    truth = np.load(out / "truth.npz")
    state = {"permx": truth["permx"], "poro": truth["poro"]}
    return simulator.simulate(MemberInput(-1, state, work))


def write_data(out: Path, case: Path) -> None:
    """Write the rate observations and the truth's noise-free rates."""
    from resdata.summary import Summary

    from apex.domain.observations import DataKey
    from apex.simulation.deck import read_summary

    rng = np.random.default_rng(SEED)
    dates = [d.isoformat()[:10] for d in Summary(str(case)).report_dates]
    keys = [DataKey(t, f"{v}:{w}") for t in dates for v, wells in WELLS.items() for w in wells]
    rates = read_summary(case, keys)
    lines = ["time,datatype,value,variance"]
    for key, rate in zip(keys, rates, strict=True):
        # A rate of 0 is observed as 0 (no member can have a negative one), with a variance of
        # its own, as the case's PET set-up.
        variance = max((RATE_RELATIVE * rate) ** 2, RATE_VARIANCE) if rate > 0 else ZERO_VARIANCE
        observed = float(rate + np.sqrt(variance) * rng.standard_normal()) if rate > 0 else 0.0
        lines.append(f"{key.time},{key.datatype},{observed!r},{variance}")
    (out / "observations.csv").write_text("\n".join(lines) + "\n")
    truth_lines = ["time,datatype,value"]
    truth_lines += [f"{k.time},{k.datatype},{float(r)!r}" for k, r in zip(keys, rates, strict=True)]
    (out / "true_rates.csv").write_text("\n".join(truth_lines) + "\n")


def main(out: Path = HERE) -> None:
    """Run the truth and write the data and the configurations."""
    with tempfile.TemporaryDirectory() as work:
        write_data(out, simulate_truth(out, Path(work)))
    write_prior(out)
    write_configs(out)
    print(f"wrote the observations and configurations to {out}")


if __name__ == "__main__":
    main()
