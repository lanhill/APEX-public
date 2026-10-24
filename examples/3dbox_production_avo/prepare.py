"""Build this example's data from the 3dBox AVO case.

The case is the APEX authors' own, set up for PET's AVO simulator, and ships here under
GPL-3.0-or-later: its PVT tables (``pvt.txt``), well schedule (``Schdl.sch``) and true model
(``truth.npz``, the log-permeability and the porosity, which ``poro.inc`` also holds for decks
run without a porosity state). This runs the truth
once with OPM Flow through APEX's simulators, and writes from it:

- ``REF.EGRID``: the grid whose active columns fix the AVO traces (``grid_file``);
- ``true_rates.csv``: the truth's noise-free rates (for ``synthetic_demo.py``'s figures);
- ``observations.csv``: the wells' rates at the 133 report times (the first of every month and
  the two vintages), with a standard deviation of
  10 % of the rate, at least the case's PET ``setup.py``'s 8 (a rate of 0 is observed as 0,
  with 10²);
- ``avo_<angle>_<vintage>.npy``: the RMS maps of each angle at each vintage, with 5 %
  noise, and ``true_response.npz``, the noise-free values (for the integration test);
- ``config_esmda.toml``, ``config_gies.toml``, ``config_esmda_noloc.toml`` and
  ``config_gies_noloc.toml`` (ES-MDA and GIES, with and without localization), whose field
  variances are the noise's.

::

    uv run python examples/3dbox_production_avo/prepare.py [CASE]

``CASE``, the case's original folder (not shipped), first refreshes the copied inputs from
it; without it the committed inputs are used. The files it writes are committed, so this
only needs rerunning to refresh them.
"""

import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
#: Seed of the observation noise.
SEED = 2061
#: The PET set-up's variance of a rate (8²), now the floor, and of a rate that is 0 (10²).
RATE_VARIANCE, ZERO_VARIANCE = 64.0, 100.0
#: A rate's relative error (standard deviation over the rate). The PET set-up's 8 alone is
#: 0.1 % of a 7000 sm3/day rate, hundreds of times narrower than the ensemble's spread: no
#: update can fit it without collapsing the ensemble.
RATE_RELATIVE = 0.10
#: The cap on ln PERMX in the deck (the PET deck's is 6, below the truth's 6.9).
PERMX_CAP = 8.0
#: The prior: members, seed, the Gaussian variogram's practical range (cells), ln PERMX's
#: and the porosity's mean and standard deviation, their correlation, and the bounds.
MEMBERS, PRIOR_SEED, PRIOR_RANGE = 100, 2026, 17.0
PERMX_PRIOR, PORO_PRIOR, RHO = (5.0, 1.0), (0.2, 0.05), 0.7
PERMX_BOUNDS, PORO_BOUNDS = (-3.0, 10.0), (0.01, 0.4)
#: The AVO maps' relative noise (standard deviation over the map's mean |value|).
AVO_NOISE = 0.05
#: The AVO vintages: the PET set-up's days 2000 and 4000. (Its baseline, day 0, is no datum.)
VINTAGES = ("2027-06-24", "2032-12-14")
ANGLES = ("10", "20")
WELLS = {"WOPR": ("PRO1", "PRO2", "PRO3"), "WWPR": ("PRO1", "PRO2", "PRO3"),
         "WWIR": ("INJ1", "INJ2", "INJ3")}  # fmt: skip


CONFIG = """# The 3dBox AVO case, written by prepare.py: {title} on the wells' rates
# and RMS maps of two angle stacks at two vintages, on OPM Flow; the log-permeability and
# the porosity are uncertain (AVO data are sensitive to porosity).
[run]
name = "3dbox_production_avo_{label}"
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
{fields}
# The AVO maps wavelet-compressed (the case studies' default for AVO data), one transform per
# angle stack: Daubechies-2 at 3 levels, hard-thresholded at the noise level estimated from
# the finest band; the methods see only the kept coefficients.
[[data_transforms]]
name = "wavelet"
options = {{ datatype = "avo_10", wavelet = "db2", level = 3 }}

[[data_transforms]]
name = "wavelet"
options = {{ datatype = "avo_20", wavelet = "db2", level = 3 }}

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

{method}
[geophysics]
pem = {{ name = "standard" }}
grid_file = "REF.EGRID"
# Overburden (lithostatic) pressure per cell, in Pa: p_ob = surface + gradient * depth,
# with depth the cell centre's true vertical depth (m, positive down, the run's INIT DEPTH).
# Rock above the reservoir with an average density of 2300 kg/m3 gives
#   gradient = 2300 kg/m3 * 9.81 m/s2 = 22,600 Pa/m (about 1 psi/ft),
# so a cell at 2400 m has p_ob = 101,325 + 22,600 * 2400 = 54.3 MPa. (The case's PET set-up uses a
# constant 320 bar; a file gives it: {{ file = "overburden.npz", key = "obvalues", unit = "bar" }}.)
overburden_pressure = {{ gradient = 22600.0, surface = 101325.0 }}   # Pa/m, Pa

[[geophysics.data]]
kind = "avo"
datatype = "avo"
vintages = {vintages}

[geophysics.data.options]   # the case's PET settings
angles = [10, 20]
reflectivity = "approx_zoeppritz_pp"
wavelet = {{ frequency = 45.0, length = 0.05 }}                 # Ricker
overburden = {{ vp = 2450.0, vs = 849.0, density = 2300.0 }}    # the PET set-up's shale
times = {{ start = 1.9224, end = 2.2150, step = 0.001 }}         # s, two-way
"""


#: Adaptive localization from the prior's correlations (Luo & Bhakta 2020), with the fixed
#: threshold 3/sqrt(N): without it the ensemble collapses, and the universal threshold
#: (about 0.45 for 18,000 cells and 100 members) keeps almost nothing.
LOCALIZATION = (
    "# Adaptive localization with the fixed threshold 3/sqrt(N) (0.3 at 100 members): without\n"
    "# it the ensemble collapses; the universal threshold keeps almost nothing here.\n"
    'localization = {{ name = "adaptive_prior", options = {{ threshold = "fixed" }} }}\n'
)
ESMDA = (
    '[method]\nname = "esmda"\noptions = {{ inflation = [4, 4, 4, 4] }}\n'
    'strategy = {{ name = "approx", options = {{ energy = 0.98 }} }}\n'
)
GIES = '[method]\nname = "gies"\nstrategy = {{ name = "rlmmac", options = {{ energy = 0.98 }} }}\n'
GIES_STOP = (
    "# Two small improvements in a row, not one, end the run (a damped first step is small).\n"
    "schedule = {{ confirm_convergence = true }}\nmax_iterations = 10\n"
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


def write_config(out: Path, variances: dict[str, float] | None = None) -> Path:
    """Write ``config_<method>.toml`` for each method; return the ES-MDA one's path.

    Each field's variance comes from ``variances`` (else 1).
    """
    fields = "".join(
        f'\n[[observations.fields]]\ntime = "{v}"\ndatatype = "avo_{a}"\n'
        f'file = "avo_{a}_{v}.npy"\nvariance = {(variances or {}).get(f"avo_{a}@{v}", 1.0)!r}\n'
        for v in VINTAGES
        for a in ANGLES
    )
    vintages = "[" + ", ".join(f'"{v}"' for v in VINTAGES) + "]"
    for label, title, method in METHODS:
        text = CONFIG.format(fields=fields, vintages=vintages, permx_cap=PERMX_CAP,
                             label=label, title=title, method=method.format())  # fmt: skip
        (out / f"config_{label}.toml").write_text(text)
    return out / "config_esmda.toml"


def copy_inputs(source: Path, out: Path) -> None:
    """Copy the case's tables and schedule; write the true model from its truth."""
    shutil.copy2(source / "include" / "pvt.txt", out / "pvt.txt")
    shutil.copy2(source / "avo" / "grid" / "Schdl.sch", out / "Schdl.sch")
    truth = np.load(source / "avo" / "true_property_values.npz")
    np.savez(out / "truth.npz", permx=truth["permx"], poro=np.round(truth["poro"], 4))
    values = "\n".join(f"{p:.4f}" for p in truth["poro"])
    (out / "poro.inc").write_text(
        f"-- The case's true porosity (true_property_values.npz)\nPORO\n{values}\n/\n"
    )


def simulate_truth(
    out: Path, work: Path, permx_cap: float = PERMX_CAP, deck: str = "deck.mako"
) -> Path:
    """Run the true model with OPM Flow in ``work``; return its case path.

    ``permx_cap = 6`` and ``deck = "deck_pet.mako"`` (the PET deck's schedule: 10 steps of
    400 days) are what SimulatorWrap's stored references were run with
    (``tests/integration/test_geophysics_3dbox.py``).
    """
    from apex.core.simulation import MemberInput
    from apex.simulation.opm_flow import OPMFlowSimulator

    simulator = OPMFlowSimulator(
        deck_template_file=str(out / deck), case_dir=str(out),
        template_values={"permx_cap": permx_cap},
    )  # fmt: skip
    truth = np.load(out / "truth.npz")
    state = {"permx": truth["permx"], "poro": truth["poro"]}
    return simulator.simulate(MemberInput(-1, state, work))


def write_data(out: Path, case: Path) -> None:
    """Write the reference grid, the rate observations and the AVO maps from the truth."""
    from resdata.summary import Summary

    from apex.domain.observations import DataKey
    from apex.services import composition
    from apex.simulation.deck import read_summary

    shutil.copy2(case.with_suffix(".EGRID"), out / "REF.EGRID")
    rng = np.random.default_rng(SEED)
    dates = [d.isoformat()[:10] for d in Summary(str(case)).report_dates]
    keys = [DataKey(t, f"{v}:{w}") for t in dates for v, wells in WELLS.items() for w in wells]
    rates = read_summary(case, keys)
    lines = ["time,datatype,value,variance"]
    for key, rate in zip(keys, rates, strict=True):
        # As the case's PET setup.py: a rate of 0 is observed as 0 (no member can have a negative
        # one), with a variance of its own.
        variance = max((RATE_RELATIVE * rate) ** 2, RATE_VARIANCE) if rate > 0 else ZERO_VARIANCE
        observed = float(rate + np.sqrt(variance) * rng.standard_normal()) if rate > 0 else 0.0
        lines.append(f"{key.time},{key.datatype},{observed!r},{variance}")
    (out / "observations.csv").write_text("\n".join(lines) + "\n")
    # The noise-free rates, for the demonstration's figures (synthetic_demo.py).
    truth_lines = ["time,datatype,value"]
    truth_lines += [f"{k.time},{k.datatype},{float(r)!r}" for k, r in zip(keys, rates, strict=True)]
    (out / "true_rates.csv").write_text("\n".join(truth_lines) + "\n")

    model = composition.build_geophysics(composition.load_config(write_config(out)))
    assert model is not None
    response: dict[str, np.ndarray] = {}
    variances: dict[str, float] = {}
    fields = sorted((n, v) for n, vs in model.field_vintages().items() for v in vs)
    for k, (name, vintage) in enumerate(fields):
        n = model.field_sizes()[name]
        values = model.forecast(case, [DataKey(vintage, name, i) for i in range(n)])
        response[f"{name}@{vintage}"] = values
        std = AVO_NOISE * float(np.abs(values).mean())
        variances[f"{name}@{vintage}"] = std**2
        # A stream per field, keyed k + 1: [SEED, 0] would repeat the rates' (SeedSequence
        # pads its entropy with zeros, so it is default_rng(SEED)).
        noise = np.random.default_rng([SEED, k + 1]).standard_normal(n)
        np.save(out / f"{name}_{vintage}.npy", values + std * noise)
    np.savez(out / "true_response.npz", **response)
    write_config(out, variances)


def main(source: Path | None = None, out: Path = HERE) -> None:
    """Refresh the inputs from ``source`` (if given), run the truth and write the data."""
    if source is not None:
        copy_inputs(source, out)
    write_prior(out)
    with tempfile.TemporaryDirectory() as work:
        write_data(out, simulate_truth(out, Path(work)))
    print(f"wrote the truth, observations and AVO maps to {out}")


if __name__ == "__main__":
    if len(sys.argv) > 2:
        raise SystemExit(__doc__)
    main(Path(sys.argv[1]) if len(sys.argv) == 2 else None)
