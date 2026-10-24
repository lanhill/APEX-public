"""Generate CHOP's MATLAB reference: ``tests/data/reference/chop/chop_step.npz``.

Draws a small case (a linear forward model), runs the author's CHOP code once on it
(RLM_MAC_CHOP.m's per-member-taper model update, and ``update_hyper.m`` /
``update_hyper_no_loc.m``), and stores the inputs (``in_*``, MATLAB layout: a mean column
last) and the outputs (``out_*``, members only).

Backends: ``transcription`` (default; ``transcription.py``, a line-by-line NumPy
transcription, used because this host's MATLAB licence was refused on 2026-10-10, error
-39,147) or ``matlab`` (the author's code itself, through ``chop_reference.m``)::

    python tools/chop_reference/generate.py [--backend matlab --run-scripts DIR]

``DIR`` is the author's ``5SpotExample/run_scripts`` (with ``autoAdaLoc/``).
"""

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from scipy.io import loadmat, savemat

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
#: The author's MATLAB (``5SpotExample/run_scripts``), for ``--backend matlab``; from
#: ``$APEX_CHOP_SCRIPTS`` (it is not in this repository).
DEFAULT_SCRIPTS = Path(os.environ.get("APEX_CHOP_SCRIPTS", "run_scripts"))
SEED = 20261010


def case(rng: np.random.Generator) -> dict[str, np.ndarray]:
    """Return the inputs: 8 model rows, 30 data (one length scale each), 15 members.

    More data than members, and energy 0.9: the truncation keeps fewer directions than
    there are data. Data far from the forecasts (10 times the model's scale) make large
    steps, so the floor binds in both hyper-parameter updates.
    """
    nm, nd, ne = 8, 30, 15
    forward = rng.standard_normal((nd, nm))
    models = rng.standard_normal((nm, ne))
    scales = rng.uniform(0.23, 0.43, (nd, ne))  # the paper's prior
    members = np.vstack([models, scales])
    ensemble = np.hstack([members, members.mean(axis=1, keepdims=True)])  # mean column
    sim = forward @ models + 0.05 * rng.standard_normal((nd, ne))
    sim_data = np.hstack([sim, forward @ models.mean(axis=1, keepdims=True)])
    observed = 10 * forward @ rng.standard_normal(nm)  # far: large steps
    perturbed = observed[:, None] + rng.standard_normal((nd, ne))
    updated = models + 0.3 * rng.standard_normal((nm, ne))  # a stand-in for m^{i+1}
    sim_new = forward @ updated + 0.05 * rng.standard_normal((nd, ne))
    sim_data_new = np.hstack([sim_new, forward @ updated.mean(axis=1, keepdims=True)])
    return {
        "ensemble": ensemble, "simData": sim_data, "perturbedData": perturbed,
        "simDataNew": sim_data_new, "lambda": np.array(0.8), "tsvdData": np.array(0.9),
        "nm": np.array(nm, dtype=np.float64),
    }  # fmt: skip


def main() -> int:
    """Write the reference; return the exit code."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-scripts", type=Path, default=DEFAULT_SCRIPTS)
    parser.add_argument("--matlab", default="matlab")
    parser.add_argument("--backend", choices=("transcription", "matlab"), default="transcription")
    args = parser.parse_args()
    inputs = case(np.random.default_rng(SEED))
    out = matlab(inputs, args) if args.backend == "matlab" else transcribed(inputs)
    if float(np.squeeze(out["alphaRaw"])) <= 1e-3:
        print("alpha hit MATLAB's 1e-3 floor (not ported): change the case", file=sys.stderr)
        return 1
    target = ROOT / "tests" / "data" / "reference" / "chop" / "chop_step.npz"
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        target,
        **{f"in_{k}": np.asarray(v, dtype=np.float64) for k, v in inputs.items()},
        out_model=out["modelUpdated"], out_model_corr=out["modelCorr"],
        out_model_rank=np.asarray(out["modelRank"], dtype=np.float64).squeeze(),
        out_hyper_localized=out["hyperLocalized"], out_hyper_unlocalized=out["hyperUnlocalized"],
        out_hyper_corr=out["hyperCorr"],
        meta_seed=np.array(SEED), meta_backend=np.array(args.backend),
    )  # fmt: skip
    print(f"wrote {target}")
    return 0


def transcribed(inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Run the transcription (``transcription.py``)."""
    sys.path.insert(0, str(HERE))
    import transcription as t  # the generator's own folder

    nm = int(inputs["nm"])
    lam, tsvd = float(inputs["lambda"]), float(inputs["tsvdData"])
    model = t.chop_model_update(
        inputs["ensemble"], inputs["simData"], inputs["perturbedData"], lam, tsvd, nm
    )
    hyper = inputs["ensemble"][nm:]
    return {
        "modelUpdated": model["model"], "modelCorr": model["corr"], "modelRank": model["rank"],
        "alphaRaw": model["alpha"],
        "hyperLocalized": t.update_hyper(hyper, inputs["simDataNew"], inputs["perturbedData"],
                                         lam, tsvd, localized=True),
        "hyperUnlocalized": t.update_hyper(hyper, inputs["simDataNew"], inputs["perturbedData"],
                                           lam, tsvd, localized=False),
        "hyperCorr": t.get_corr_mtx(hyper[:, :-1], inputs["simDataNew"][:, :-1]),
    }  # fmt: skip


def matlab(inputs: dict[str, np.ndarray], args: argparse.Namespace) -> dict[str, np.ndarray]:
    """Run the author's MATLAB code (``chop_reference.m``)."""
    with tempfile.TemporaryDirectory() as scratch:
        work = Path(scratch)
        savemat(work / "in.mat", inputs)
        paths = f"addpath('{HERE}'); addpath('{args.run_scripts}'); " + (
            f"addpath('{args.run_scripts / 'autoAdaLoc'}'); "
        )
        command = paths + f"chop_reference('{work / 'in.mat'}', '{work / 'out.mat'}')"
        subprocess.run([args.matlab, "-batch", command], check=True)
        out: dict[str, np.ndarray] = loadmat(work / "out.mat")
    return out


if __name__ == "__main__":
    sys.exit(main())
