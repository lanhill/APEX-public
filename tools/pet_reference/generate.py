"""Generate PET reference outputs for APEX parity tests.

Run with a PET interpreter, never the APEX environment, and clear
``PYTHONPATH`` so that no other PET checkout shadows the one being used::

    env -u PYTHONPATH .venv-pet/bin/python tools/pet_reference/generate.py --list
    env -u PYTHONPATH .venv-pet/bin/python tools/pet_reference/generate.py [<case> ...]
    env -u PYTHONPATH .venv-pet-fork/bin/python tools/pet_reference/generate.py [<case> ...]

Each case belongs to one PET source (``SOURCES``): upstream PET, or PET-fork for GIES.
A run generates only the cases of the source the interpreter provides, and refuses cases
of another source.

Each case writes ``tests/data/reference/pet/<case>.npz`` holding the seeded inputs
(``in_*``), PET's outputs (``out_*``) and provenance (``meta_*``). To add a case, write a
function that builds inputs from the given generator, runs the PET method on them and
returns the arrays, then decorate it with ``@case("<name>", seed=<int>, source=<name>)``.
Inputs are injected into PET explicitly so that APEX can reproduce the deterministic step
exactly. PET is imported inside the case functions, because the sources have different
layouts.

This script is outside the ``apex`` package: it is the only place allowed to import PET.
"""

import argparse
import importlib.util
import json
import os
import subprocess
import sys
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from types import SimpleNamespace

import numpy as np

#: PET sources the references are generated from (see docs/migration/pet_parity.md).
SOURCES = {
    "upstream": "e65fb1058411c8dad7ca70a09d501e0b18075e75",  # Python-Ensemble-Toolbox/PET
    "fork": "1348861b08314b8b3f3e1ccf886c7c095273e202",  # not public (GIES)
}


def _pet_source_or_exit() -> tuple[Path, str]:
    """Return the root and commit of the PET code this interpreter imports.

    Runs before any PET import. A ``PYTHONPATH`` entry pointing at another checkout (which
    may carry its own ``PET.egg-info``) would silently change the code the references are
    generated from, so it is refused outright. The commit comes from a ``.pet_commit``
    marker next to ``pipt`` (source exported with ``git archive``) or from the installed
    distribution's ``direct_url.json``.
    """
    if os.environ.get("PYTHONPATH"):
        sys.exit("PYTHONPATH is set; run with `env -u PYTHONPATH` so only the chosen PET is used")
    spec = importlib.util.find_spec("pipt")
    if spec is None or spec.origin is None:
        sys.exit(
            "PET is not importable here; use .venv-pet or .venv-pet-fork "
            "(see tests/data/reference/README.md)"
        )
    root = Path(spec.origin).resolve().parents[1]
    misc = importlib.util.find_spec("misc")
    if misc is None or misc.origin is None or not Path(misc.origin).resolve().is_relative_to(root):
        sys.exit(f"misc does not resolve inside the PET source at {root}")
    marker = root / ".pet_commit"
    if marker.is_file():
        return root, marker.read_text().strip()
    try:
        dist = metadata.distribution("PET")
    except metadata.PackageNotFoundError:
        dist = None
    if dist is not None and Path(str(dist.locate_file(""))).resolve() == root:
        direct_url = dist.read_text("direct_url.json")
        commit = json.loads(direct_url).get("vcs_info", {}).get("commit_id") if direct_url else None
        if commit:
            return root, str(commit)
    result = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    )
    return root, result.stdout.strip() if result.returncode == 0 else "unknown"


PET_ROOT, PET_COMMIT = _pet_source_or_exit()

OUT_DIR = Path(__file__).resolve().parents[2] / "tests" / "data" / "reference" / "pet"

CaseFunction = Callable[[np.random.Generator], dict[str, np.ndarray]]


@dataclass(frozen=True)
class Case:
    """A registered reference case."""

    seed: int
    source: str
    function: CaseFunction


CASES: dict[str, Case] = {}


def case(name: str, seed: int, source: str = "upstream") -> Callable[[CaseFunction], CaseFunction]:
    """Register a reference case under ``name``, generated with ``seed`` from ``source``."""
    if source not in SOURCES:
        raise ValueError(f"unknown PET source {source!r}; expected one of {sorted(SOURCES)}")

    def decorator(function: CaseFunction) -> CaseFunction:
        if name in CASES:
            raise ValueError(f"reference case {name!r} is already defined")
        CASES[name] = Case(seed, source, function)
        return function

    return decorator


class InjectedNormals:
    """Stand-in for PET's RandomState that returns pre-drawn standard normals.

    PET draws observation perturbations with ``rng.randn(n_data, n_ensemble)``. Handing it
    this object instead makes PET use exactly the ``z`` stored in the reference file.
    """

    def __init__(self, z: np.ndarray) -> None:
        self._z = z

    def randn(self, *shape: int) -> np.ndarray:
        """Return the injected normals, checking PET asked for the expected shape."""
        if shape != self._z.shape:
            raise ValueError(f"PET asked for normals of shape {shape}, injected {self._z.shape}")
        return self._z.copy()


def _pet_approx_step(
    state: np.ndarray,
    predicted: np.ndarray,
    perturbed: np.ndarray,
    cov_data: np.ndarray,
    scale_data: np.ndarray,
    state_std: np.ndarray,
    energy: float,
) -> np.ndarray:
    """Call PET's ``approx_update`` through the smallest scheme object it reads from."""
    from pipt.update_schemes.analysis.approx import approx_update

    ne = state.shape[1]
    scheme = SimpleNamespace(
        cov_data=cov_data,
        scale_data=scale_data,
        state_scaling=state_std,
        proj=(np.eye(ne) - np.ones((ne, ne)) / ne) / np.sqrt(ne - 1),  # ensemble_base.py
        keys_da={},  # no emp_cov
        trunc_energy=energy,
        lam=0,  # ES-MDA sets lambda to 0
        localization=SimpleNamespace(name=None),  # no localization
        logger=None,
    )
    return approx_update(scheme).update(enX=state, enY=predicted, enE=perturbed).step


def _esmda_inputs(
    rng: np.random.Generator, n_state: int, n_data: int, n_ensemble: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # State rows with very different scales exercise PET's state scaling, which must
    # cancel. PET's layout is (n, n_ensemble), the same as APEX's, so nothing is transposed.
    state_std = rng.uniform(0.1, 10.0, n_state)
    state = rng.standard_normal((n_state, n_ensemble)) * state_std[:, None]
    forward = rng.standard_normal((n_data, n_state)) / state_std[None, :]
    predicted = np.tanh(forward @ state)  # mildly nonlinear forward model
    observations = rng.standard_normal(n_data)
    z = rng.standard_normal((n_data, n_ensemble))
    return state, state_std, predicted, observations, z


@case("esmda_approx_diagonal", seed=20260925)
def esmda_approx_diagonal(rng: np.random.Generator) -> dict[str, np.ndarray]:
    """One ES-MDA step (``alpha = 4``) with PET's approx analysis and a diagonal ``C_d``.

    Mirrors ``ESMDA.calc_analysis``: perturb with ``gen_real(obs, alpha * C_d)``, then
    ``approx_update`` with the returned factor as ``scale_data``. Prior, forward-model
    output and ``z`` are injected. PET's default energy 0.98 keeps all 8 singular values
    for this spectrum, so truncation is exercised by ``esmda_approx_truncated``.
    """
    from misc.sampling import gen_real

    state, state_std, predicted, observations, z = _esmda_inputs(rng, 20, 8, 50)
    variances = rng.uniform(0.05, 0.5, 8)
    alpha, energy = 4.0, 0.98
    perturbed, scale_data = gen_real(
        observations, alpha * variances, 50, rng=InjectedNormals(z), return_chol=True
    )
    step = _pet_approx_step(state, predicted, perturbed, variances, scale_data, state_std, energy)
    return {
        "in_state": state,
        "in_predicted": predicted,
        "in_observations": observations,
        "in_variances": variances,
        "in_z": z,
        "in_alpha": np.array(alpha),
        "in_energy": np.array(energy),
        "out_perturbed_observations": perturbed,
        "out_posterior": state + step,
    }


@case("es_approx_step", seed=20261010)
def es_approx_step(rng: np.random.Generator) -> dict[str, np.ndarray]:
    """One ES update with PET's approx analysis.

    Mirrors ``EnKF.calc_analysis``, which ES inherits: ``gen_real(obs, C_d)`` with the
    un-inflated ``C_d`` and ``approx_update`` at ``lam = 0``. Prior, forward-model output
    and ``z`` are injected; energy 0.98 (PET's approx default). The ``ES`` class itself is
    not constructed: its wiring to this call was checked by reading ``es.py``/``enkf.py``.
    """
    from misc.sampling import gen_real

    state, state_std, predicted, observations, z = _esmda_inputs(rng, 20, 12, 30)
    variances = rng.uniform(0.05, 0.5, 12)
    energy = 0.98
    perturbed, scale_data = gen_real(
        observations, variances, 30, rng=InjectedNormals(z), return_chol=True
    )
    step = _pet_approx_step(state, predicted, perturbed, variances, scale_data, state_std, energy)
    return {
        "in_state": state,
        "in_predicted": predicted,
        "in_observations": observations,
        "in_variances": variances,
        "in_z": z,
        "in_energy": np.array(energy),
        "out_perturbed_observations": perturbed,
        "out_posterior": state + step,
    }


@case("lmenrml_approx_step", seed=20261008)
def lmenrml_approx_step(rng: np.random.Generator) -> dict[str, np.ndarray]:
    """One LM-EnRML step (``lambda = 7.3``) with PET's approx analysis.

    Mirrors ``IterativeEnRML``: the perturbed observations are drawn once from the
    un-inflated ``C_d`` (``gen_real(obs, C_d)``), ``scale_data`` is its factor, and
    ``approx_update`` runs with the scheme's ``lam``. 30 data and 25 members at PET's
    LM-EnRML energy 0.95 exercise the truncation. Prior, forward-model output and ``z`` are
    injected.
    """
    from misc.sampling import gen_real

    state, state_std, predicted, observations, z = _esmda_inputs(rng, 20, 30, 25)
    variances = rng.uniform(0.05, 0.5, 30)
    lam, energy = 7.3, 0.95
    perturbed, scale_data = gen_real(
        observations, variances, 25, rng=InjectedNormals(z), return_chol=True
    )
    from pipt.update_schemes.analysis.approx import approx_update

    scheme = SimpleNamespace(
        cov_data=variances,
        scale_data=scale_data,
        state_scaling=state_std,
        proj=(np.eye(25) - np.ones((25, 25)) / 25) / np.sqrt(24),  # ensemble_base.py
        keys_da={},  # no emp_cov
        trunc_energy=energy,
        lam=lam,
        localization=SimpleNamespace(name=None),  # no localization
        logger=None,
    )
    step = approx_update(scheme).update(enX=state, enY=predicted, enE=perturbed).step
    return {
        "in_state": state,
        "in_predicted": predicted,
        "in_observations": observations,
        "in_variances": variances,
        "in_z": z,
        "in_lambda": np.array(lam),
        "in_energy": np.array(energy),
        "out_perturbed_observations": perturbed,
        "out_posterior": state + step,
    }


@case("lmenrml_full_step", seed=20261011)
def lmenrml_full_step(rng: np.random.Generator) -> dict[str, np.ndarray]:
    """One LM step with PET's full analysis: the data term plus the prior-increment term.

    ``full_update`` runs through the smallest scheme it reads: ``state_scaling`` is PET's
    ``calc_scaling`` for two variables (12 and 8 rows, prior variances 4 and 0.09, as a
    ``prior_info`` would give), ``Am`` is left ``None`` so that ``ext_Am`` builds it from
    ``prior_enX``, and the iterate ``enX`` has moved away from the prior so that the prior
    term is not zero. 20 states and 15 members make ``Am`` rank-deficient (truncated at
    0.95). Prior, iterate, forecast and ``z`` are injected.
    """
    from misc.sampling import gen_real
    from pipt.update_schemes.analysis.full import full_update

    n_state, n_data, ne = 20, 10, 15
    scaling = np.concatenate([np.full(12, np.sqrt(4.0)), np.full(8, np.sqrt(0.09))])
    prior = rng.standard_normal((n_state, ne)) * scaling[:, None]
    state = prior + 0.3 * rng.standard_normal((n_state, ne)) * scaling[:, None]  # an iterate
    forward = rng.standard_normal((n_data, n_state)) / scaling[None, :]
    predicted = np.tanh(forward @ state)
    observations = rng.standard_normal(n_data)
    z = rng.standard_normal((n_data, ne))
    variances = rng.uniform(0.05, 0.5, n_data)
    lam, energy = 3.0, 0.95
    perturbed, scale_data = gen_real(
        observations, variances, ne, rng=InjectedNormals(z), return_chol=True
    )
    scheme = SimpleNamespace(
        cov_data=variances,
        scale_data=scale_data,
        state_scaling=scaling,
        proj=(np.eye(ne) - np.ones((ne, ne)) / ne) / np.sqrt(ne - 1),  # ensemble_base.py
        prior_enX=prior,
        Am=None,
        keys_da={},
        trunc_energy=energy,
        lam=lam,
        localization=SimpleNamespace(name=None),
        logger=None,
    )
    step = full_update(scheme).update(enX=state, enY=predicted, enE=perturbed, prior=prior).step
    return {
        "in_prior": prior,
        "in_state": state,
        "in_predicted": predicted,
        "in_observations": observations,
        "in_variances": variances,
        "in_z": z,
        "in_scaling": scaling,
        "in_lambda": np.array(lam),
        "in_energy": np.array(energy),
        "out_perturbed_observations": perturbed,
        "out_am": scheme.Am,
        "out_step": step,
    }


class _PetLog:
    """What LMEnRML's damping hooks log to: callable, with ``info``."""

    def __call__(self, *args: object) -> None:
        pass

    def info(self, *args: object) -> None:
        pass


#: Scripted misfit sequences for PET's LMEnRML.score_and_commit: per scenario the prior's
#: member misfits and then each trial's, chosen to reach every branch of the damping rules.
_LM_SCENARIOS = {
    # improved with a smaller spread (lambda / 5), improved with a larger spread (lambda
    # kept), rejected (lambda * 5), improved again, then within the tolerance (converged).
    "mixed": ([100, 120, 80], [[60, 70, 50], [40, 60, 20], [55, 45, 50], [30, 35, 25],
                               [29.8, 35.1, 24.9]]),
    # a worse trial within the tolerance: converged, and not taken.
    "worse_converged": ([100, 110, 90], [[100.5, 110.2, 90.1]]),
    # lambda at lambda_max: stop, taking the trial only if it improved.
    "lambda_max": ([100, 110, 90], [[150, 160, 140], [150, 170, 130], [80, 90, 70]]),
    # at lambda_min: improved with a smaller spread, lambda not reduced further.
    "lambda_min": ([100, 120, 80], [[50, 55, 45]]),
}  # fmt: skip
_LM_SETTINGS = {  # scenario -> (lambda, lambda_factor, lambda_min, lambda_max, tolerance)
    "mixed": (100.0, 5.0, 0.01, 1e10, 0.01),
    "worse_converged": (100.0, 5.0, 0.01, 1e10, 0.01),
    "lambda_max": (100.0, 5.0, 0.01, 2500.0, 0.01),
    "lambda_min": (0.01, 5.0, 0.01, 1e10, 0.01),
}


@case("lmenrml_schedule", seed=20261009)
def lmenrml_schedule(rng: np.random.Generator) -> dict[str, np.ndarray]:
    """PET's LM-EnRML damping rules: ``LMEnRML.score_and_commit`` on scripted misfits.

    An ``LMEnRML`` is made with ``object.__new__`` and given only what
    ``score_and_commit`` and the damping hooks read; ``score`` returns the scripted member
    misfits. Per trial the reference holds the lambda it ran with, whether it was accepted,
    whether the run converged, and the lambda after. ``lambda = 'auto'`` is resolved by
    PET's own ``LMEnRML.score`` on a random prior (``out_auto_lambda``).
    """
    from pipt.update_schemes.enrml import LMEnRML

    arrays: dict[str, np.ndarray] = {}
    for name, (prior, trials) in _LM_SCENARIOS.items():
        lam, factor, lam_min, lam_max, tol = _LM_SETTINGS[name]
        scheme = object.__new__(LMEnRML)
        scheme.lam, scheme.lam_factor, scheme.lam_min, scheme.lam_max = (
            lam,
            factor,
            lam_min,
            lam_max,
        )
        scheme.data_misfit_tol, scheme.logger, scheme.iteration = tol, _PetLog(), 0
        scheme.data_misfit_mean = scheme.prior_data_misfit_mean = float(np.mean(prior))
        scheme.data_misfit_std = float(np.std(prior))
        scheme.ensemble_misfit = np.asarray(prior, dtype=float)
        rows = []
        for trial in trials:
            scheme.score = lambda pred_data=None, t=trial: np.asarray(t, dtype=float)  # type: ignore[method-assign]
            used = scheme.lam
            scheme.score_and_commit()
            rows.append([used, float(scheme.step_accepted), float(scheme._converged), scheme.lam])
            if scheme._converged:
                break
        arrays[f"in_{name}_prior"] = np.asarray(prior, dtype=float)
        arrays[f"in_{name}_trials"] = np.asarray(trials, dtype=float)
        arrays[f"in_{name}_settings"] = np.asarray(_LM_SETTINGS[name])
        # Per trial: lambda used, accepted, converged, lambda after.
        arrays[f"out_{name}"] = np.asarray(rows)
    # lambda = 'auto': 0.5 * mean(prior misfit) / n_data, from PET's LMEnRML.score.
    variances = rng.uniform(0.05, 0.5, 12)
    perturbed = rng.standard_normal((12, 30))
    predicted = rng.standard_normal((12, 30))
    auto = object.__new__(LMEnRML)
    auto.lam, auto.enObs, auto.cov_data = "auto", perturbed, variances
    auto.score(predicted)
    arrays.update(
        in_auto_perturbed=perturbed,
        in_auto_predicted=predicted,
        in_auto_variances=variances,
        out_auto_lambda=np.array(auto.lam),
    )
    return arrays


@case("subspace_steps", seed=20261012)
def subspace_steps(rng: np.random.Generator) -> dict[str, np.ndarray]:
    """Two iterations of PET's subspace analysis, as GN-EnRML runs it (``step_scale``).

    ``subspace_update`` runs through the smallest scheme it reads. Iteration 0 starts from
    ``current_W = 0`` (and sets ``E``); its weights ``W = gamma * w_step`` are committed,
    as ``score_and_commit`` does on acceptance, and iteration 1 steps from them, so the
    ``W``-dependent terms (``Omega``, ``deltaM``) are exercised. The states are PET's
    ``propose_state`` weight branch. ``lambda = 1.5`` (an LM-damped step) and ``gamma =
    0.4`` (a GN step length) are both active. 20 data for 12 members: ``S`` has a null
    singular value, dropped by the 0.95 energy truncation. Prior, forecasts and ``z`` are
    injected; the forward model is ``tanh(G x)``.
    """
    from misc.sampling import gen_real
    from pipt.update_schemes.analysis.subspace import subspace_update

    n_state, n_data, ne = 15, 20, 12
    prior = rng.standard_normal((n_state, ne)) * rng.uniform(0.5, 3.0, n_state)[:, None]
    forward = rng.standard_normal((n_data, n_state)) / 3.0
    observations = rng.standard_normal(n_data)
    z = rng.standard_normal((n_data, ne))
    variances = rng.uniform(0.05, 0.5, n_data)
    lam, gamma, energy = 1.5, 0.4, 0.95
    perturbed, scale_data = gen_real(
        observations, variances, ne, rng=InjectedNormals(z), return_chol=True
    )
    proj = (np.eye(ne) - np.ones((ne, ne)) / ne) / np.sqrt(ne - 1)  # ensemble_base.py
    scheme = SimpleNamespace(
        scale_data=scale_data, proj=proj, iteration=0, trunc_energy=energy, lam=lam
    )
    analysis = subspace_update(scheme)
    out: dict[str, np.ndarray] = {}
    state = prior
    for k in (0, 1):
        predicted = np.tanh(forward @ state)
        scheme.iteration = k
        w_step = analysis.update(enX=state, enY=predicted, enE=perturbed).w_step
        weights = scheme.current_W + gamma * w_step  # propose_state
        state = prior @ (np.eye(ne) + weights / np.sqrt(ne - 1))
        scheme.current_W = weights  # score_and_commit on acceptance
        out[f"out_predicted_{k}"] = predicted
        out[f"out_weights_{k}"] = weights
        out[f"out_state_{k}"] = state
    return {
        "in_prior": prior,
        "in_forward": forward,
        "in_observations": observations,
        "in_variances": variances,
        "in_z": z,
        "in_lambda": np.array(lam),
        "in_gamma": np.array(gamma),
        "in_energy": np.array(energy),
        "out_perturbed_observations": perturbed,
        **out,
    }


@case("subspace_dense", seed=20261014)
def subspace_dense(rng: np.random.Generator) -> dict[str, np.ndarray]:
    """One iteration of PET's subspace analysis with a correlated ``C_d`` (no truncation).

    PET's ``gen_real`` returns the upper Cholesky factor ``U`` of ``C_d = U^T U`` and the
    analysis whitens with ``solve(U, .)``, i.e. ``U^-1`` (the ES-MDA whitening bug). The
    whitening cancels in the subspace step only when ``S`` spans the data space (``n_data
    <= N - 1``, no truncation): 8 data, 15 members, ``energy = 1`` (``out_weights``).
    Otherwise the kept subspace depends on it: the same with ``energy = 0.8``
    (``out_weights_truncated``), and 30 data for 12 members with only the null singular
    value dropped (``wide``).
    ``lambda = 0.5``.
    """
    from misc.sampling import gen_real
    from pipt.update_schemes.analysis.subspace import subspace_update

    n_state, n_data, ne = 10, 8, 15
    prior = rng.standard_normal((n_state, ne))
    forward = rng.standard_normal((n_data, n_state)) / 2.0
    predicted = np.tanh(forward @ prior)
    observations = rng.standard_normal(n_data)
    z = rng.standard_normal((n_data, ne))
    a = rng.standard_normal((n_data, n_data))
    covariance = 0.05 * (a @ a.T + n_data * np.eye(n_data))
    lam, energy = 0.5, 1.0
    perturbed, scale_data = gen_real(
        observations, covariance, ne, rng=InjectedNormals(z), return_chol=True
    )
    proj = (np.eye(ne) - np.ones((ne, ne)) / ne) / np.sqrt(ne - 1)
    scheme = SimpleNamespace(
        scale_data=scale_data, proj=proj, iteration=0, trunc_energy=energy, lam=lam
    )
    w_step = subspace_update(scheme).update(enX=prior, enY=predicted, enE=perturbed).w_step
    scheme.trunc_energy = 0.8
    truncated = subspace_update(scheme).update(enX=prior, enY=predicted, enE=perturbed).w_step
    wide: dict[str, np.ndarray] = {}
    n_wide, ne_wide = 30, 12
    wide["in_wide_prior"] = rng.standard_normal((n_state, ne_wide))
    wide["in_wide_predicted"] = np.tanh(
        rng.standard_normal((n_wide, n_state)) / 2.0 @ wide["in_wide_prior"]
    )
    wide["in_wide_observations"] = rng.standard_normal(n_wide)
    wide["in_wide_z"] = rng.standard_normal((n_wide, ne_wide))
    b = rng.standard_normal((n_wide, n_wide))
    wide["in_wide_covariance"] = 0.05 * (b @ b.T + n_wide * np.eye(n_wide))
    wide_perturbed, wide_scale = gen_real(
        wide["in_wide_observations"],
        wide["in_wide_covariance"],
        ne_wide,
        rng=InjectedNormals(wide["in_wide_z"]),
        return_chol=True,
    )
    wide_scheme = SimpleNamespace(
        scale_data=wide_scale,
        proj=(np.eye(ne_wide) - np.ones((ne_wide, ne_wide)) / ne_wide) / np.sqrt(ne_wide - 1),
        iteration=0,
        # All but S's null singular value (rank N - 1 < n_data), which PET keeps at 1.
        trunc_energy=1.0 - 1e-9,
        lam=lam,
    )
    wide["out_wide_perturbed_observations"] = wide_perturbed
    wide["out_wide_scale_data"] = np.asarray(wide_scale)
    wide["out_wide_weights"] = (
        subspace_update(wide_scheme)
        .update(enX=wide["in_wide_prior"], enY=wide["in_wide_predicted"], enE=wide_perturbed)
        .w_step
    )
    return {
        **wide,
        "in_prior": prior,
        "in_predicted": predicted,
        "in_observations": observations,
        "in_covariance": covariance,
        "in_z": z,
        "in_lambda": np.array(lam),
        "in_energy": np.array(energy),
        "out_perturbed_observations": perturbed,
        "out_scale_data": np.asarray(scale_data),
        "out_weights": w_step,  # W = 0 + 1 * w_step
        "out_weights_truncated": truncated,
    }


@case("esmda_prior_term_linear", seed=20261015)
def esmda_prior_term_linear(rng: np.random.Generator) -> dict[str, np.ndarray]:
    """PET's ES-MDA with the full and the subspace analysis on a linear-Gaussian case.

    Each step is what ``ESMDA.calc_analysis`` does: redraw the perturbations with
    ``alpha * C_d`` (``gen_real``, injected ``z``), then run the analysis with ``lam = 0``
    and the *original* prior (``prior = prior_enX``); the subspace analysis's weights are
    committed after each step (``score_and_commit``), and its ``E`` is the step's
    perturbations (``calc_analysis`` sets it). The forward model is linear, ``g(x) = G x``,
    with 3 unknowns, 8 data, 2000 members and ``alpha = [4, 4, 4, 4]``; energy 1 (the
    subspace analysis: all but the null singular values). The test
    compares the posterior spread with the exact posterior (see
    docs/migration/pet_parity.md: re-imposing the prior at every step is not ES-MDA).
    """
    from misc.sampling import gen_real
    from pipt.update_schemes.analysis.full import full_update
    from pipt.update_schemes.analysis.subspace import subspace_update

    n_state, n_data, ne, prior_std = 3, 8, 2000, 0.5
    forward = rng.standard_normal((n_data, n_state))
    variances = rng.uniform(0.05, 0.2, n_data)
    observations = forward @ np.array([0.5, -1.0, 0.8]) + np.sqrt(variances) * rng.standard_normal(
        n_data
    )
    prior = prior_std * rng.standard_normal((n_state, ne))
    alphas = [4.0, 4.0, 4.0, 4.0]
    zs = [rng.standard_normal((n_data, ne)) for _ in alphas]
    proj = (np.eye(ne) - np.ones((ne, ne)) / ne) / np.sqrt(ne - 1)
    out: dict[str, np.ndarray] = {}
    for flavour in ("full", "subspace"):
        state = prior.copy()
        scheme = SimpleNamespace(
            state_scaling=np.full(n_state, prior_std),  # calc_scaling: sqrt(prior variance)
            proj=proj,
            prior_enX=prior,
            Am=None,
            keys_da={},
            # S has rank 3 < 8 data here: the subspace analysis drops only its null singular
            # values (PET keeps them at energy 1 and divides by rounding noise).
            trunc_energy=1.0 if flavour == "full" else 1.0 - 1e-9,
            lam=0,  # ESMDA.__init__
            localization=SimpleNamespace(name=None),
            logger=None,
            iteration=0,
        )
        for k, (alpha, z) in enumerate(zip(alphas, zs, strict=True)):
            perturbed, scale_data = gen_real(
                observations, alpha * variances, ne, rng=InjectedNormals(z), return_chol=True
            )
            scheme.cov_data, scheme.scale_data = alpha * variances, scale_data
            scheme.E, scheme.iteration = perturbed @ proj, k
            predicted = forward @ state
            if flavour == "full":
                step = (
                    full_update(scheme)
                    .update(enX=state, enY=predicted, enE=perturbed, prior=prior)
                    .step
                )
                state = state + step
            else:
                w_step = (
                    subspace_update(scheme).update(enX=state, enY=predicted, enE=perturbed).w_step
                )
                scheme.current_W = scheme.current_W + w_step  # propose_state, then commit
                state = prior @ (np.eye(ne) + scheme.current_W / np.sqrt(ne - 1))
        out[f"out_posterior_{flavour}"] = state
    return {
        "in_forward": forward,
        "in_variances": variances,
        "in_observations": observations,
        "in_prior": prior,
        "in_prior_std": np.array(prior_std),
        "in_alphas": np.array(alphas),
        "in_z": np.stack(zs),
        **out,
    }


def _transform_steps(
    rng: np.random.Generator, analysis: str, dense: bool = False
) -> dict[str, np.ndarray]:
    """Two iterations of PET's ``subspace2`` or ``margis`` (the matrix transform, W_0 = I).

    As ``subspace_steps``: iteration 0 from ``current_W = I``, its ``W = I + gamma * W_step``
    committed, iteration 1 from it; the states are ``propose_state``'s matrix branch,
    ``mean(prior) + prior @ proj * sqrt(ne - 1) @ W``. ``lambda = 1.5``, ``gamma = 0.4``,
    15 states, 20 data in three types (8, 7 and 5 rows) for 12 members. ``margis`` reads
    the types from ``data_layout.row_datatypes()``. ``dense`` uses a correlated ``C_d``
    (one iteration is enough to pin PET's ``U^-1`` whitening).
    """
    from misc.sampling import gen_real
    from pipt.update_schemes.analysis.margis import margIS_update
    from pipt.update_schemes.analysis.subspace2 import subspace2_update

    n_state, n_data, ne = 15, 20, 12
    prior = rng.standard_normal((n_state, ne)) * rng.uniform(0.5, 3.0, n_state)[:, None]
    forward = rng.standard_normal((n_data, n_state)) / 3.0
    observations = rng.standard_normal(n_data)
    z = rng.standard_normal((n_data, ne))
    if dense:
        a = rng.standard_normal((n_data, n_data))
        covariance = 0.05 * (a @ a.T + n_data * np.eye(n_data))
    else:
        covariance = rng.uniform(0.05, 0.5, n_data)
    types = np.array(["A"] * 8 + ["B"] * 7 + ["C"] * 5)
    lam, gamma = 1.5, 0.4
    perturbed, scale_data = gen_real(
        observations, covariance, ne, rng=InjectedNormals(z), return_chol=True
    )
    proj = (np.eye(ne) - np.ones((ne, ne)) / ne) / np.sqrt(ne - 1)
    scheme = SimpleNamespace(
        scale_data=scale_data,
        proj=proj,
        iteration=0,
        lam=lam,
        ne=ne,
        data_layout=SimpleNamespace(row_datatypes=lambda: types),
    )
    update = (subspace2_update if analysis == "subspace2" else margIS_update)(scheme)
    out: dict[str, np.ndarray] = {}
    state = prior
    for k in (0,) if dense else (0, 1):
        predicted = np.tanh(forward @ state)
        scheme.iteration = k
        w_step = update.update(enX=state, enY=predicted, enE=perturbed).W_step
        weights = scheme.current_W + gamma * w_step  # propose_state
        state = np.mean(prior, axis=1, keepdims=True) + prior @ proj * np.sqrt(ne - 1) @ weights
        scheme.current_W = weights  # score_and_commit on acceptance
        out[f"out_predicted_{k}"] = predicted
        out[f"out_weights_{k}"] = weights
        out[f"out_state_{k}"] = state
    return {
        "in_prior": prior,
        "in_forward": forward,
        "in_observations": observations,
        "in_covariance": covariance,
        "in_z": z,
        "in_types": types,
        "in_lambda": np.array(lam),
        "in_gamma": np.array(gamma),
        "out_perturbed_observations": perturbed,
        "out_scale_data": np.asarray(scale_data),
        **out,
    }


@case("subspace2_steps", seed=20261016)
def subspace2_steps(rng: np.random.Generator) -> dict[str, np.ndarray]:
    """Two iterations of PET's ``subspace2`` (see ``_transform_steps``)."""
    return _transform_steps(rng, "subspace2")


@case("margis_steps", seed=20261017)
def margis_steps(rng: np.random.Generator) -> dict[str, np.ndarray]:
    """Two iterations of PET's ``margis``, three data types (see ``_transform_steps``)."""
    return _transform_steps(rng, "margis")


@case("subspace2_dense", seed=20261018)
def subspace2_dense(rng: np.random.Generator) -> dict[str, np.ndarray]:
    """One iteration of PET's ``subspace2`` with a correlated ``C_d`` (``U^-1`` whitening)."""
    return _transform_steps(rng, "subspace2", dense=True)


def _simulatorwrap_eclipse(upscale: dict[str, object]) -> object:
    """Return a SimulatorWrap ``eclipse`` holding only ``upscale``, with ``_Haar`` runnable.

    SimulatorWrap's ``_Haar`` slices with ``image.shape[0] / 2``, a float on Python 3; the
    shim recompiles its own source with ``//`` in those two lines and nothing else.
    """
    import inspect
    import textwrap

    from subsurface.multphaseflow.eclipse import eclipse

    source = textwrap.dedent(inspect.getsource(eclipse._Haar))
    for name in ("image", "weights"):
        source = source.replace(
            f"{name}.shape[0] / 2, :{name}.shape[1] / 2",
            f"{name}.shape[0] // 2, :{name}.shape[1] // 2",
        )
    fixed = source
    assert fixed.count("// 2") == 4, "SimulatorWrap's _Haar changed; revisit the shim"
    namespace: dict[str, object] = {"np": np}
    exec(compile(fixed, "<simulatorwrap _Haar, integer halves>", "exec"), namespace)
    instance = object.__new__(eclipse)
    instance.upscale = upscale
    instance._Haar = namespace["_Haar"].__get__(instance)  # type: ignore[attr-defined]
    return instance


#: Upscaling cases: (method, dim (rows, cols), max_trunc, max_diff, wells [x, y], radii).
_UPSCALING = {
    "unif_half": ("unif", (16, 12), 0.5, 0.0, [[3, 4], [10, 13]], [1, 2]),
    "unif_all": ("unif", (13, 10), 1.0, 0.0, [[5, 5]], [1]),
    "unif_none": ("unif", (8, 8), 0.0, 0.0, [], []),
    "haar_dyadic": ("haar", (16, 16), 0.6, 0.15, [[4, 4]], [1]),
    "haar_odd": ("haar", (12, 10), 0.6, 0.15, [], []),
}


@case("upscaling_records", seed=20261019)
def upscaling_records(rng: np.random.Generator) -> dict[str, np.ndarray]:
    """SimulatorWrap's grid coarsening (``subsurface`` @ ``d29dc57``): ``COARSEN`` records.

    Per case, SimulatorWrap's ``_unif``/``_Haar`` and ``write_coarse`` run on a field that
    is smooth and low in one part (merged) and has a high, rough channel (kept fine),
    with wells and radii. ``_Haar`` runs through ``_simulatorwrap_eclipse``'s shim.
    ``write_coarse`` stores the records in ``self.coarse`` (``[i1, i2, j1, j2, 1, 1, 1, 1,
    1]``, 1-based).
    """
    out: dict[str, np.ndarray] = {}
    for name, (method, dim, max_trunc, max_diff, wells, radius) in _UPSCALING.items():
        rows, cols = dim
        field = 1.0 + 0.02 * rng.standard_normal((rows, cols))
        field[:, cols // 2 :] += 5.0 + rng.uniform(0.0, 3.0, (rows, cols - cols // 2))
        upscale = {
            "maxtrunc": max_trunc,
            "maxdiff": max_diff,
            "wells": wells,
            "radius": radius,
            "dim": list(dim),
        }
        sim = _simulatorwrap_eclipse(upscale)
        weights = sim._unif(dim) if method == "unif" else sim._Haar(field.copy(), np.ones(dim))  # type: ignore[attr-defined]
        sim.write_coarse("", weights, field)  # type: ignore[attr-defined]
        out[f"in_{name}_field"] = field.reshape(-1)
        out[f"out_{name}_records"] = np.asarray(sim.coarse, dtype=np.int64).reshape(-1, 9)  # type: ignore[attr-defined]
    out["in_simulatorwrap_commit"] = np.array("d29dc57")  # the upscaling code's source
    return out


@case("hybrid_step", seed=20261020)
def hybrid_step(rng: np.random.Generator) -> dict[str, np.ndarray]:
    """One multilevel ES-MDA step with PET's ``hybrid_update`` (two fidelity levels).

    ``hybrid_update`` runs through the smallest scheme it reads: per level ``proj`` and
    ``scale_data`` (from ``gen_real`` with ``alpha * C_d``, injected ``z`` per level, as
    ``esmda_hybrid.calc_analysis`` draws them), ``state_scaling``, ``multilevel['ml_weights']``,
    ``trunc_energy`` and ``lam = 0``. Levels of 10 and 25 members, weights 0.3 and 0.7, 12
    states, 15 data (level 0 truncated by the energy 0.95), ``alpha = 3``. The level-0
    forecast is a biased, noisier version of the level-1 model (a coarse model).
    """
    from misc.sampling import gen_real
    from pipt.update_schemes.analysis.hybrid import hybrid_update

    n_state, n_data, sizes, weights = 12, 15, (10, 25), (0.3, 0.7)
    alpha, energy = 3.0, 0.95
    state_std = rng.uniform(0.5, 2.0, n_state)
    forward = rng.standard_normal((n_data, n_state)) / 3.0
    observations = rng.standard_normal(n_data)
    variances = rng.uniform(0.05, 0.3, n_data)
    states, predicted, zs, perturbed, scales, projs = [], [], [], [], [], []
    for level, n in enumerate(sizes):
        x = rng.standard_normal((n_state, n)) * state_std[:, None]
        y = np.tanh(forward @ x)
        if level == 0:  # the coarse model: biased and noisier
            y = y + 0.1 + 0.05 * rng.standard_normal(y.shape)
        z = rng.standard_normal((n_data, n))
        d, scale = gen_real(
            observations, alpha * variances, n, rng=InjectedNormals(z), return_chol=True
        )
        states.append(x)
        predicted.append(y)
        zs.append(z)
        perturbed.append(d)
        scales.append(scale)
        projs.append((np.eye(n) - np.ones((n, n)) / n) / np.sqrt(n - 1))
    scheme = SimpleNamespace(
        proj=projs,
        scale_data=scales,
        state_scaling=state_std,
        tot_level=len(sizes),
        keys_da={},
        trunc_energy=energy,
        lam=0,
        multilevel={"ml_weights": np.asarray(weights)},
    )
    steps = hybrid_update(scheme).update(enX=states, enY=predicted, enE=perturbed).step
    out: dict[str, np.ndarray] = {
        "in_sizes": np.asarray(sizes),
        "in_weights": np.asarray(weights),
        "in_alpha": np.array(alpha),
        "in_energy": np.array(energy),
        "in_observations": observations,
        "in_variances": variances,
        "in_state_scaling": state_std,
    }
    for level in range(len(sizes)):
        out[f"in_state_{level}"] = states[level]
        out[f"in_predicted_{level}"] = predicted[level]
        out[f"in_z_{level}"] = zs[level]
        out[f"out_perturbed_{level}"] = perturbed[level]
        out[f"out_step_{level}"] = steps[level]
    return out


@case("approx_emp_cov", seed=20261021)
def approx_emp_cov(rng: np.random.Generator) -> dict[str, np.ndarray]:
    """PET's approx analysis with ``emp_cov``: ``C_d`` replaced by the perturbations' ``E E^T``.

    ``approx_update`` with ``keys_da = {"emp_cov": True}``, ``lambda = 1.5`` and energy 0.95
    (truncating), no localization; 25 data for 15 members. Prior, forecast and ``z`` are
    injected.
    """
    from misc.sampling import gen_real
    from pipt.update_schemes.analysis.approx import approx_update

    n_state, n_data, ne = 12, 25, 15
    state, state_std, predicted, observations, z = _esmda_inputs(rng, n_state, n_data, ne)
    variances = rng.uniform(0.05, 0.5, n_data)
    lam, energy = 1.5, 0.95
    perturbed, scale_data = gen_real(
        observations, variances, ne, rng=InjectedNormals(z), return_chol=True
    )
    scheme = SimpleNamespace(
        cov_data=variances,
        scale_data=scale_data,
        state_scaling=state_std,
        proj=(np.eye(ne) - np.ones((ne, ne)) / ne) / np.sqrt(ne - 1),
        keys_da={"emp_cov": True},
        trunc_energy=energy,
        lam=lam,
        localization=SimpleNamespace(name=None),
        logger=None,
    )
    step = approx_update(scheme).update(enX=state, enY=predicted, enE=perturbed).step
    return {
        "in_state": state,
        "in_predicted": predicted,
        "in_observations": observations,
        "in_variances": variances,
        "in_z": z,
        "in_lambda": np.array(lam),
        "in_energy": np.array(energy),
        "out_perturbed_observations": perturbed,
        "out_step": step,
    }


@case("approx_adjoint", seed=20261023)
def approx_adjoint(rng: np.random.Generator) -> dict[str, np.ndarray]:
    """PET's approx analysis with adjoints (``enAdj``): ``Y = mean(enAdj) @ enX @ PI``.

    ``approx_update`` with an injected per-member sensitivity ``(n_d, n_x, N)`` that varies
    between members (a nonlinear model's Jacobians), ``lambda = 0.5``, energy 0.95.
    """
    from misc.sampling import gen_real
    from pipt.update_schemes.analysis.approx import approx_update

    n_state, n_data, ne = 12, 15, 20
    state, state_std, predicted, observations, z = _esmda_inputs(rng, n_state, n_data, ne)
    adjoints = rng.standard_normal((n_data, n_state, 1)) + 0.3 * rng.standard_normal(
        (n_data, n_state, ne)
    )
    variances = rng.uniform(0.05, 0.5, n_data)
    lam, energy = 0.5, 0.95
    perturbed, scale_data = gen_real(
        observations, variances, ne, rng=InjectedNormals(z), return_chol=True
    )
    scheme = SimpleNamespace(
        cov_data=variances,
        scale_data=scale_data,
        state_scaling=state_std,
        proj=(np.eye(ne) - np.ones((ne, ne)) / ne) / np.sqrt(ne - 1),
        keys_da={},
        trunc_energy=energy,
        lam=lam,
        localization=SimpleNamespace(name=None),
        logger=None,
    )
    step = (
        approx_update(scheme).update(enX=state, enY=predicted, enE=perturbed, enAdj=adjoints).step
    )
    return {
        "in_state": state,
        "in_predicted": predicted,
        "in_adjoints": adjoints,
        "in_observations": observations,
        "in_variances": variances,
        "in_z": z,
        "in_lambda": np.array(lam),
        "in_energy": np.array(energy),
        "out_perturbed_observations": perturbed,
        "out_step": step,
    }


class _LogCapture:
    """A logger for PET's QAQC that keeps every message."""

    def __init__(self) -> None:
        self.messages: list[str] = []

    def info(self, message: object) -> None:
        self.messages.append(str(message))


def _scores_from_log(messages: list[str], marker: str) -> list[float]:
    """Return the ``Score: x`` values of the first message containing ``marker``, in order."""
    import re

    text = next(m for m in messages if marker in m)
    return [float(v) for v in re.findall(r"Score: ([-+0-9.eE]+)", text)]


@case("qaqc_diagnostics", seed=20261022)
def qaqc_diagnostics(rng: np.random.Generator) -> dict[str, np.ndarray]:
    """PET's QA/QC (``qaqc_tools.QAQC``): Kalman gains, Mahalanobis scores, movement.

    A ``QAQC`` is made with ``object.__new__`` and given the adapted frames it reads
    (``en_fcst``, ``en_obs``, ``en_var``, ``en_time`` per data type, the state, the prior,
    ``lam``). Three data types (6, 6 and 4 report points), 30 members, a 20-cell field and
    a scalar. Outputs: ``_gain`` per (type, parameter) exactly; the level-1 and level-2
    Mahalanobis scores, and the level-1 scores of data types projected on their leading
    component, as PET logs them (4 significant digits); ``calc_da_stat``'s log.
    """
    from pipt.misc_tools.qaqc_tools import QAQC

    n = 30
    types = {"WOPR:P1": 6, "WWCT:P1": 6, "WBHP:I1": 4}
    prior = {"permx": rng.standard_normal((20, n)), "mult": rng.standard_normal((1, n))}
    state = {k: v + 0.3 * rng.standard_normal(v.shape) + 0.2 for k, v in prior.items()}
    qaqc = object.__new__(QAQC)
    qaqc.logger = _LogCapture()
    qaqc.data_types = list(types)
    qaqc.en_fcst, qaqc.en_obs, qaqc.en_var, qaqc.en_time = {}, {}, {}, {}
    qaqc.en_fcst_vec, qaqc.en_obs_vec, qaqc.en_var_vec = {}, {}, {}
    out: dict[str, np.ndarray] = {}
    for k, (typ, n_t) in enumerate(types.items()):
        g = rng.standard_normal((n_t, 20)) / 5
        fcst = g @ state["permx"] + 0.5 * state["mult"] + 0.1 * rng.standard_normal((n_t, n))
        qaqc.en_fcst[typ] = fcst
        qaqc.en_obs[typ] = (fcst.mean(axis=1) + rng.normal(0, 0.5, n_t))[:, None]
        qaqc.en_var[typ] = rng.uniform(0.05, 0.2, n_t)[:, None]
        qaqc.en_time[typ] = list(range(n_t))
        out[f"in_forecast_{k}"] = fcst
        out[f"in_observations_{k}"] = qaqc.en_obs[typ][:, 0]
        out[f"in_variances_{k}"] = qaqc.en_var[typ][:, 0]
    qaqc.pred_data, qaqc.state, qaqc.ini_state = object(), state, prior
    qaqc.list_state, qaqc.ne, qaqc.lam, qaqc.prior_info = list(state), n, 0.0, {}
    qaqc.localization = None
    qaqc._crossplots = lambda *args: None  # no figures here
    for k, typ in enumerate(types):
        for name in state:
            fcst, obs, var = qaqc._lumped(typ)
            gain = qaqc._gain(name, fcst, obs, var, False)
            out[f"out_gain_{k}_{name}"] = np.asarray(gain, dtype=np.float64)
            qaqc.lam = 0.7  # the LM form; PET's code is (1 + lam) C_dd + C_d
            gain = qaqc._gain(name, fcst, obs, var, False)
            out[f"out_gain_damped_{k}_{name}"] = np.asarray(gain, dtype=np.float64)
            qaqc.lam = 0.0
    qaqc.calc_mahalanobis((1, None, 2, None, 1, "time"))
    messages = qaqc.logger.messages
    level1 = [m for m in messages if "Largest values are" in m]
    out["out_level1_scores"] = np.array(_scores_from_log([level1[0]], "Largest"))
    out["out_level2_scores"] = np.array(_scores_from_log(messages, "level-2"))
    out["out_level1_type_scores"] = np.array(_scores_from_log([level1[1]], "Largest"))
    qaqc.calc_da_stat()
    out["out_da_stat_log"] = np.array(qaqc.logger.messages[-1])
    for name in state:
        out[f"in_prior_{name}"] = prior[name]
        out[f"in_state_{name}"] = state[name]
    return out


#: Scripted misfit sequences for PET's GNEnRML.score_and_commit (as ``_LM_SCENARIOS``).
_GN_SCENARIOS = {
    # improved with a smaller spread (gamma relaxes), improved with a larger spread (kept),
    # rejected (gamma / factor), improved with a smaller spread again (relaxes less: the
    # exponent grows with the accepted iterations), then within the tolerance.
    "mixed": ([100, 120, 80], [[60, 70, 50], [40, 60, 20], [55, 45, 50], [30, 35, 25],
                               [29.8, 35.1, 24.9]]),
    # a worse trial within the tolerance: converged, and not taken.
    "worse_converged": ([100, 110, 90], [[100.5, 110.2, 90.1]]),
    # gamma_factor <= 1: the step length never changes.
    "fixed": ([100, 120, 80], [[60, 70, 50], [80, 90, 70], [40, 45, 35]]),
}  # fmt: skip
_GN_SETTINGS = {  # scenario -> (gamma, gamma_max, gamma_factor, tolerance)
    "mixed": (0.2, 0.5, 2.5, 0.01),
    "worse_converged": (0.2, 0.5, 2.5, 0.01),
    "fixed": (0.3, 0.5, 1.0, 0.01),
}


#: The configurations of ``approx_autoadaloc_rankr``: (name, threshold, cutoff, PET type,
#: lambda, emp_cov, shuffle seed). The defaults first, then each threshold and taper.
_AUTOADALOC_RUNS = (
    ("adaptive_hard", "adaptive", 0.3, "hard", 0.0, False, 101),
    ("adaptive_soft", "adaptive", 0.3, "soft", 0.0, False, 102),
    ("adaptive_sigmoid", "adaptive", 0.3, "sigm", 0.0, False, 103),
    ("fixed_hard", "fixed", 0.4, "hard", 0.0, False, 104),
    ("universal_soft", "universal", 0.3, "soft", 0.0, False, 105),
    ("universal_hard_lm", "universal", 0.3, "hard", 1.5, False, 106),
    ("adaptive_hard_emp_cov", "adaptive", 0.3, "hard", 1.5, True, 107),
)


@case("approx_autoadaloc_rankr", seed=20261024)
def approx_autoadaloc_rankr(rng: np.random.Generator) -> dict[str, np.ndarray]:
    """Upstream PET's ``autoadaloc`` in the approx analysis, ``projection = "rank-r"``.

    ``approx_update`` with ``AutoAdaptiveLocalization`` (``auto_ada_loc.py``) for each
    configuration of ``_AUTOADALOC_RUNS``: two parameter blocks of 30 rows (``parameters``
    and ``field = [30]``), 25 data, 20 members, energy 0.95 (truncating). Prior, forecast and
    ``z`` are injected; the shuffle draws from ``np.random.default_rng(<shuffle seed>)``,
    passed to PET as the localization's ``rng``, so APEX draws the same permutation from a
    generator with that seed (``out_permutation_<name>`` checks it). Layouts are PET's and
    APEX's ``(n, n_ensemble)``.
    """
    from misc.sampling import gen_real
    from pipt.localization.auto_ada_loc import AutoAdaptiveLocalization
    from pipt.update_schemes.analysis.approx import approx_update

    block, n_data, ne, energy = 30, 25, 20, 0.95
    n_state = 2 * block
    state, state_std, predicted, observations, z = _esmda_inputs(rng, n_state, n_data, ne)
    variances = rng.uniform(0.05, 0.5, n_data)
    perturbed, scale_data = gen_real(
        observations, variances, ne, rng=InjectedNormals(z), return_chol=True
    )
    out: dict[str, np.ndarray] = {
        "in_state": state,
        "in_predicted": predicted,
        "in_observations": observations,
        "in_variances": variances,
        "in_z": z,
        "in_energy": np.array(energy),
        "in_block_sizes": np.array([block, block]),
        "out_perturbed_observations": perturbed,
    }
    for name, threshold, cutoff, taper, lam, emp_cov, shuffle_seed in _AUTOADALOC_RUNS:
        localization = AutoAdaptiveLocalization(
            {
                "field": [block],
                "parameters": ["P1", "P2"],
                "threshold": threshold,
                "cutoff": cutoff,
                "type": taper,
                "projection": "rank-r",
            },
            rng=np.random.default_rng(shuffle_seed),
        )
        scheme = SimpleNamespace(
            cov_data=variances,
            scale_data=scale_data,
            state_scaling=state_std,
            proj=(np.eye(ne) - np.ones((ne, ne)) / ne) / np.sqrt(ne - 1),
            keys_da={"emp_cov": emp_cov},
            trunc_energy=energy,
            lam=lam,
            localization=localization,
            logger=None,
        )
        step = approx_update(scheme).update(enX=state, enY=predicted, enE=perturbed).step
        out[f"in_threshold_{name}"] = np.array(threshold)
        out[f"in_cutoff_{name}"] = np.array(cutoff)
        out[f"in_taper_{name}"] = np.array(taper)
        out[f"in_emp_cov_{name}"] = np.array(emp_cov)
        out[f"in_lambda_{name}"] = np.array(lam)
        out[f"in_shuffle_seed_{name}"] = np.array(shuffle_seed)
        out[f"out_permutation_{name}"] = np.random.default_rng(shuffle_seed).permutation(ne)
        out[f"out_step_{name}"] = step
    return out


#: ``distance_entries_masks``: the grid in APEX's Eclipse terms [nx, ny, nz]; PET's field
#: is [nz, ny, nx] (its x is Eclipse j, its y is i), so both flatten to the same order.
_DIST_GRID = (10, 8, 2)
_DIST_TIMES = ("1", "2")
#: APEX data type -> PET column; SEIS is a vector datum (3 components per time).
_DIST_TYPES = {"WOPR:PRO1": "WOPR PRO1", "WWCT:PRO2": "WWCT PRO2", "BHP:INJ1": "BHP INJ1",
               "SEIS": "SEIS"}  # fmt: skip
_DIST_SEIS = 3
#: Entries in APEX's form (1-based Eclipse positions); a later one replaces an earlier one.
_DIST_ENTRIES: tuple[dict[str, object], ...] = (
    {"data": "WOPR:PRO1", "time": "*", "parameter": "PERMX", "taper": "gaspari_cohn",
     "position": [3, 2, 1], "radius": 3, "z_range": "all", "anisotropy": [1.0, 0.0]},
    {"data": "WWCT:PRO2", "time": "*", "parameter": "*", "taper": "furrer_bengtsson",
     "position": [8, 6, 2], "radius": 4, "z_range": 0, "anisotropy": [2.0, 30.0]},
    {"data": "BHP:INJ1", "time": "1", "parameter": "PORO", "taper": "region",
     "position": [5, 5, 1], "z_range": 1},
    {"data": "SEIS", "time": "2", "parameter": "*", "taper": "gaspari_cohn",
     "position": [9, 2, 2], "radius": 2, "z_range": 0, "anisotropy": [1.5, -45.0]},
    {"data": "WOPR:PRO1", "time": "2", "parameter": "PERMX", "taper": "gaspari_cohn",
     "position": [1, 8, 1], "radius": 5, "z_range": "all"},
    {"data": "BHP:INJ1", "time": "2", "parameter": "PERMX", "taper": "gaspari_cohn",
     "position": [6, 3, 2], "radius": 12, "z_range": "all", "anisotropy": [1.0, 15.0]},
)  # fmt: skip
_PET_TAPERS = {"gaspari_cohn": "gc", "furrer_bengtsson": "fb", "region": "region"}


def _pet_distance_entry(entry: dict[str, object]) -> dict[str, object]:
    """Return an entry in PET's ``entries`` dict form (0-based x = j - 1, y = i - 1)."""
    i, j, k = (int(v) for v in entry["position"])  # type: ignore[attr-defined]
    ratio, angle = entry.get("anisotropy", (1.0, 0.0))  # type: ignore[misc]
    z_range = entry.get("z_range", "all")
    return {
        "taper": _PET_TAPERS[str(entry["taper"])],
        "x": j - 1, "y": i - 1, "z": k - 1,
        "radius": int(entry.get("radius", 1)),  # type: ignore[call-overload]
        "z_range": ":" if z_range == "all" else int(z_range),  # type: ignore[call-overload]
        "aniso": float(ratio), "rotation": float(angle),
        "data_type": "*" if entry["data"] == "*" else _DIST_TYPES[str(entry["data"])],
        "time": "*" if entry["time"] == "*" else float(str(entry["time"])),
        "param": str(entry["parameter"]),
    }  # fmt: skip


@case("distance_entries_masks", seed=20261025)
def distance_entries_masks(rng: np.random.Generator) -> dict[str, np.ndarray]:
    """Upstream PET's ``distance_loc``: its taper and the approx step it localizes.

    ``DistanceLocalization`` (``distance_localization.py``) with the ``entries`` of
    ``_DIST_ENTRIES`` (each kernel, anisotropy and rotation, ``z_range`` all or a layer
    range, wildcards, a later entry replacing an earlier one, a radius beyond the grid, a
    vector datum, and data and variables no entry pairs) on a 10 x 8 x 2 grid with two
    parameters, once over all cells and once over the active cells of an ``actnum``. The
    taper is PET's sparse operator, densified; the steps are ``approx_update``'s
    ``distance_loc`` branch at lambda 0 and 1.5 (energy 0.95), prior, forecast and ``z``
    injected. PET's layout ``(n, N)`` is APEX's; its rows (``[nz, ny, nx]`` C order) are
    Eclipse order and its columns time-major, as APEX's keys here.
    """
    import pandas as pd
    from misc.sampling import gen_real
    from pipt.localization.distance_localization import DistanceLocalization
    from pipt.update_schemes.analysis.approx import approx_update

    nx, ny, nz = _DIST_GRID
    cells, ne, energy = nx * ny * nz, 20, 0.95
    columns = list(_DIST_TYPES.values())
    frame = pd.DataFrame(
        [[[0.0] * _DIST_SEIS if c == "SEIS" else 0.0 for c in columns] for _ in _DIST_TIMES],
        index=[float(t) for t in _DIST_TIMES],
        columns=columns,
        dtype=object,
    )
    n_data = len(_DIST_TIMES) * (len(columns) - 1 + _DIST_SEIS)
    active = rng.uniform(size=cells) > 0.2
    out: dict[str, np.ndarray] = {
        "in_grid": np.array(_DIST_GRID),
        "in_entries": np.array(json.dumps(_DIST_ENTRIES)),
        "in_times": np.array(_DIST_TIMES),
        "in_datatypes": np.array(list(_DIST_TYPES)),
        "in_seis_components": np.array(_DIST_SEIS),
        "in_actnum": active,
        "in_energy": np.array(energy),
        "in_n_ensemble": np.array(ne),
    }
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        actnum_file = Path(tmp) / "actnum.npz"
        np.savez(actnum_file, actnum=active)
        for layout, n_cells in (("full", cells), ("actnum", int(active.sum()))):
            info: dict[str, object] = {
                "field": [nz, ny, nx],
                "entries": [_pet_distance_entry(e) for e in _DIST_ENTRIES],
            }
            if layout == "actnum":
                info["actnum"] = str(actnum_file)
            localization = DistanceLocalization(
                info,
                data=frame,
                parameters=["PERMX", "PORO"],
                ensemble_size=ne,
                prior_info={p: {"nx": nx, "ny": ny, "nz": nz} for p in ("PERMX", "PORO")},
            )
            taper = localization().toarray()
            n_state = 2 * n_cells
            state, state_std, predicted, observations, z = _esmda_inputs(rng, n_state, n_data, ne)
            variances = rng.uniform(0.05, 0.5, n_data)
            perturbed, scale_data = gen_real(
                observations, variances, ne, rng=InjectedNormals(z), return_chol=True
            )
            out[f"out_taper_{layout}"] = taper
            out[f"in_state_{layout}"] = state
            out[f"in_predicted_{layout}"] = predicted
            out[f"in_observations_{layout}"] = observations
            out[f"in_variances_{layout}"] = variances
            out[f"in_z_{layout}"] = z
            out[f"out_perturbed_observations_{layout}"] = perturbed
            for lam in (0.0, 1.5):
                scheme = SimpleNamespace(
                    cov_data=variances,
                    scale_data=scale_data,
                    state_scaling=state_std,
                    proj=(np.eye(ne) - np.ones((ne, ne)) / ne) / np.sqrt(ne - 1),
                    keys_da={},
                    trunc_energy=energy,
                    lam=lam,
                    localization=localization,
                    logger=None,
                )
                step = approx_update(scheme).update(enX=state, enY=predicted, enE=perturbed)
                out[f"out_step_{layout}_lambda_{lam:g}"] = np.asarray(step.step)
    return out


@case("distance_entries_miscentred", seed=20261026)
def distance_entries_miscentred(rng: np.random.Generator) -> dict[str, np.ndarray]:
    """Upstream PET's ``distance_loc`` taper where its stencil crop is asymmetric (a PET defect).

    One Gaspari-Cohn entry, radius 14, anisotropy ratio 0.5 rotated 45 degrees, on a 12 x 26
    single-layer grid (Eclipse ``nx = 12``, ``ny = 26``; PET field ``[1, 26, 12]``), centred
    at Eclipse ``(i, j) = (7, 14)``. Its support reaches past PET's ``2 n`` stencil on one
    side only, so ``_crop_kernel`` trims it asymmetrically and ``_place_kernel`` centres it
    one cell off. Stored to document the shift; nothing is drawn from ``rng``.
    """
    import pandas as pd
    from pipt.localization.distance_localization import DistanceLocalization

    nx, ny, i, j = 12, 26, 7, 14
    entry = {
        "taper": "gc", "x": j - 1, "y": i - 1, "z": 0, "radius": 14, "z_range": ":",
        "aniso": 0.5, "rotation": 45.0, "data_type": "d", "time": 1.0, "param": "p",
    }  # fmt: skip
    localization = DistanceLocalization(
        {"field": [1, ny, nx], "entries": [entry]},
        data=pd.DataFrame([[0.0]], index=[1.0], columns=["D"]),
        parameters=["P"],
        ensemble_size=20,
        prior_info={"P": {"nx": nx, "ny": ny, "nz": 1}},
    )
    return {
        "in_grid": np.array([nx, ny, 1]),
        "in_position": np.array([i, j, 1]),
        "in_radius": np.array(14.0),
        "in_anisotropy": np.array([0.5, 45.0]),
        "out_taper": localization().toarray()[:, 0],
    }


@case("gnenrml_schedule", seed=20261013)
def gnenrml_schedule(rng: np.random.Generator) -> dict[str, np.ndarray]:
    """PET's GN-EnRML step-length rules: ``GNEnRML.score_and_commit`` on scripted misfits.

    As ``lmenrml_schedule``. ``iteration`` is advanced after every accepted trial, as the
    base loop (``scheme_base.py``) does, because ``_on_improved`` reads it. Per trial the
    reference holds the gamma it ran with, whether it was accepted, whether the run
    converged, and the gamma after. ``gamma = 'auto'`` is resolved by PET's
    ``_read_damping_options`` (``out_auto_gamma``).
    """
    from pipt.update_schemes.enrml import GNEnRML

    del rng  # deterministic: every input is scripted
    arrays: dict[str, np.ndarray] = {}
    for name, (prior, trials) in _GN_SCENARIOS.items():
        gamma, gamma_max, factor, tol = _GN_SETTINGS[name]
        scheme = object.__new__(GNEnRML)
        scheme._read_damping_options(
            {"gamma": gamma, "gamma_max": gamma_max, "gamma_factor": factor}
        )
        scheme.data_misfit_tol, scheme.logger, scheme.iteration = tol, _PetLog(), 0
        scheme.data_misfit_mean = scheme.prior_data_misfit_mean = float(np.mean(prior))
        scheme.data_misfit_std = float(np.std(prior))
        scheme.ensemble_misfit = np.asarray(prior, dtype=float)
        rows = []
        for trial in trials:
            scheme.score = lambda pred_data=None, t=trial: np.asarray(t, dtype=float)  # type: ignore[method-assign]
            used = scheme.gamma
            scheme.score_and_commit()
            rows.append([used, float(scheme.step_accepted), float(scheme._converged), scheme.gamma])
            if scheme._converged:
                break
            if scheme.step_accepted:
                scheme.iteration += 1
        arrays[f"in_{name}_prior"] = np.asarray(prior, dtype=float)
        arrays[f"in_{name}_trials"] = np.asarray(trials, dtype=float)
        arrays[f"in_{name}_settings"] = np.asarray(_GN_SETTINGS[name])
        # Per trial: gamma used, accepted, converged, gamma after.
        arrays[f"out_{name}"] = np.asarray(rows)
    auto = object.__new__(GNEnRML)
    auto._read_damping_options({"gamma": "auto"})
    arrays["out_auto_gamma"] = np.array(auto.gamma)
    return arrays


@case("esmda_approx_truncated", seed=20260927)
def esmda_approx_truncated(rng: np.random.Generator) -> dict[str, np.ndarray]:
    """As ``esmda_approx_diagonal`` but with ``n_data > n_ensemble`` and active truncation.

    This is the usual reservoir regime (rank-deficient, non-square SVD). The energy is
    placed halfway between two cumulative singular-value fractions, so that rounding
    differences of ~1e-16 cannot change the kept rank, and at least 3 values are dropped.
    PET's own ``truncSVD`` rank is stored as ``out_rank``.
    """
    import pipt.misc_tools.analysis_tools as at
    from misc.sampling import gen_real

    n_state, n_data, n_ensemble = 30, 60, 25
    state, state_std, predicted, observations, z = _esmda_inputs(rng, n_state, n_data, n_ensemble)
    variances = rng.uniform(0.05, 0.5, n_data)
    alpha = 4.0
    perturbed, scale_data = gen_real(
        observations, alpha * variances, n_ensemble, rng=InjectedNormals(z), return_chol=True
    )
    # Whitened predicted-data anomalies exactly as approx.py forms them.
    proj = (np.eye(n_ensemble) - np.ones((n_ensemble, n_ensemble)) / n_ensemble) / np.sqrt(
        n_ensemble - 1
    )
    singular_values = np.linalg.svd(predicted @ proj / scale_data[:, None], compute_uv=False)
    fractions = np.cumsum(singular_values) / singular_values.sum()
    target_rank = n_ensemble - 1 - 4  # anomalies have rank n_ensemble - 1; drop 4 more
    energy = float((fractions[target_rank - 2] + fractions[target_rank - 1]) / 2)
    margin = (fractions[target_rank - 1] - fractions[target_rank - 2]) / 2
    if margin < 1e-6:
        raise ValueError(f"energy margin {margin:.2e} too small to pin the rank")
    _, kept, _ = at.truncSVD(predicted @ proj / scale_data[:, None], energy=energy)
    step = _pet_approx_step(state, predicted, perturbed, variances, scale_data, state_std, energy)
    return {
        "in_state": state,
        "in_predicted": predicted,
        "in_observations": observations,
        "in_variances": variances,
        "in_z": z,
        "in_alpha": np.array(alpha),
        "in_energy": np.array(energy),
        "out_perturbed_observations": perturbed,
        "out_rank": np.array(kept.size),
        "out_posterior": state + step,
    }


@case("esmda_approx_dense", seed=20260926)
def esmda_approx_dense(rng: np.random.Generator) -> dict[str, np.ndarray]:
    """As ``esmda_approx_diagonal`` but with a correlated ``C_d`` and no truncation.

    PET factors ``alpha * C_d = U^T U`` (upper Cholesky) and whitens with ``U^-1``, so its
    gain uses ``U U^T`` in place of ``alpha * C_d``. APEX diverges here on purpose; the
    parity test checks that this case reproduces exactly that formula.
    """
    from misc.sampling import gen_real

    state, state_std, predicted, observations, z = _esmda_inputs(rng, 20, 8, 50)
    a = rng.standard_normal((8, 8))
    covariance = 0.05 * (a @ a.T + 8 * np.eye(8))
    alpha, energy = 4.0, 1.0
    perturbed, scale_data = gen_real(
        observations, alpha * covariance, 50, rng=InjectedNormals(z), return_chol=True
    )
    step = _pet_approx_step(state, predicted, perturbed, covariance, scale_data, state_std, energy)
    return {
        "in_state": state,
        "in_predicted": predicted,
        "in_observations": observations,
        "in_covariance": covariance,
        "in_z": z,
        "in_alpha": np.array(alpha),
        "in_energy": np.array(energy),
        "out_perturbed_observations": perturbed,
        "out_posterior": state + step,
    }


@contextmanager
def _global_normals(z: np.ndarray) -> Iterator[None]:
    """Make ``np.random.randn`` return ``z`` while PET-fork's geostat draws from it.

    The old ``Cholesky.gen_real`` takes no stream argument and calls the global
    ``np.random.randn``; patching it for the duration is the only way to inject ``z``.
    """
    original = np.random.randn
    np.random.randn = InjectedNormals(z).randn  # type: ignore[assignment]
    try:
        yield
    finally:
        np.random.randn = original  # type: ignore[assignment]


def _gies_rlmmac_case(
    rng: np.random.Generator, n_state: int, n_data: int, n_ensemble: int, dense: bool = False
) -> dict[str, np.ndarray]:
    """One RLM-MAC analysis with PET-fork's ``rlmmac_update`` and a diagonal ``C_d``.

    Mirrors ``GIESMixIn``: perturb once with ``gen_real(obs, C_d)``; forecast the members
    and the mean model (column ``ne`` of ``aug_pred_data``, as ``ensemble.py`` appends it);
    form ``pert_preddata`` exactly as ``GIESMixIn.calc_analysis`` does (copied below, since
    that method needs a full ensemble); then call ``rlmmac_update.update`` on the smallest
    object it reads from. ``energy = 1`` disables PET's truncation, whose old rule APEX does
    not reproduce. PET's layout is (n, n_ensemble), the same as APEX's.
    """
    import pipt.misc_tools.analysis_tools as at
    from pipt.geostat.decomp import Cholesky
    from pipt.update_schemes.gies.rlmmac_update import rlmmac_update

    state_std = rng.uniform(0.1, 10.0, n_state)
    state = rng.standard_normal((n_state, n_ensemble)) * state_std[:, None]
    forward = rng.standard_normal((n_data, n_state)) / state_std[None, :]
    predicted = np.tanh(forward @ state)
    reference_prediction = np.tanh(forward @ state.mean(axis=1))  # forecast of the mean model
    observations = rng.standard_normal(n_data)
    variances = rng.uniform(0.05, 0.5, n_data)
    if dense:
        a = rng.standard_normal((n_data, n_data))
        covariance = 0.05 * (a @ a.T + n_data * np.eye(n_data))
    z = rng.standard_normal((n_data, n_ensemble))
    damping = 0.5
    cov_data = covariance if dense else variances

    with _global_normals(z):
        perturbed, scale_data = Cholesky().gen_real(
            observations, cov_data, n_ensemble, return_chol=True
        )
    aug_pred_data = np.hstack([predicted, reference_prediction[:, None]])  # (nd, ne + 1)
    # GIESMixIn.calc_analysis, diagonal and matrix branches.
    if len(scale_data.shape) == 1:
        pert_preddata = np.dot(
            np.expand_dims(scale_data ** (-1), axis=1), np.ones((1, n_ensemble))
        ) * (aug_pred_data[:, 0:n_ensemble] - aug_pred_data[:, n_ensemble, None])
    else:
        from scipy.linalg import solve

        pert_preddata = solve(
            scale_data, aug_pred_data[:, 0:n_ensemble] - aug_pred_data[:, n_ensemble, None]
        )
    scheme = SimpleNamespace(
        pert_preddata=pert_preddata,
        current_state={"x": state},
        list_states=["x"],
        cell_index=None,
        trunc_energy=1.0,  # `if trunc_energy < 1` is false: no truncation
        keys_da={},  # no localization, no emp_cov
        state_scaling=state_std,
        proj=(np.eye(n_ensemble) - np.ones((n_ensemble, n_ensemble)) / n_ensemble)
        / np.sqrt(n_ensemble - 1),
        scale_data=scale_data,
        real_obs_data=perturbed,
        aug_pred_data=aug_pred_data,
        ne=n_ensemble,
        lam=damping,
    )
    rlmmac_update.update(scheme)  # the mix-in method, called with the stub as self
    arrays = {
        "in_state": state,
        "in_predicted": predicted,
        "in_reference_prediction": reference_prediction,
        "in_observations": observations,
        "in_z": z,
        "in_damping": np.array(damping),
        "out_perturbed_observations": perturbed,
        "out_posterior": state + scheme.step,
        "out_misfit": at.calc_objectivefun(perturbed, aug_pred_data, cov_data),
    }
    arrays["in_covariance" if dense else "in_variances"] = cov_data
    return arrays


@case("gies_rlmmac_diagonal", seed=20260928, source="fork")
def gies_rlmmac_diagonal(rng: np.random.Generator) -> dict[str, np.ndarray]:
    """RLM-MAC analysis with ``n_data < n_ensemble`` (see ``_gies_rlmmac_case``)."""
    return _gies_rlmmac_case(rng, n_state=20, n_data=12, n_ensemble=40)


@case("gies_rlmmac_wide", seed=20260929, source="fork")
def gies_rlmmac_wide(rng: np.random.Generator) -> dict[str, np.ndarray]:
    """RLM-MAC analysis with ``n_data > n_ensemble`` (see ``_gies_rlmmac_case``)."""
    return _gies_rlmmac_case(rng, n_state=30, n_data=60, n_ensemble=25)


@case("gies_rlmmac_dense", seed=20260930, source="fork")
def gies_rlmmac_dense(rng: np.random.Generator) -> dict[str, np.ndarray]:
    """RLM-MAC with a correlated ``C_d``: PET-fork whitens with the upper factor ``U^-1``.

    Its step is then ``dX dYu^T (dYu dYu^T + lambda U U^T)^-1 (D - Y)`` with ``C_d = U^T U``;
    the parity test pins that, and checks that APEX uses ``C_d`` itself.
    """
    return _gies_rlmmac_case(rng, n_state=20, n_data=12, n_ensemble=40, dense=True)


@contextmanager
def _injected_permutation(permutation: np.ndarray) -> Iterator[None]:
    """Make ``np.random.permutation(n)`` return ``permutation`` (PET-fork's shuffle)."""
    original = np.random.permutation

    def fixed(n: int) -> np.ndarray:
        if n != permutation.size:
            raise ValueError(f"PET asked for a permutation of {n}, injected {permutation.size}")
        return permutation.copy()

    np.random.permutation = fixed  # type: ignore[assignment]
    try:
        yield
    finally:
        np.random.permutation = original  # type: ignore[assignment]


def _derangement(rng: np.random.Generator, n: int) -> np.ndarray:
    """Draw a permutation without fixed points, which PET's fix-up code leaves unchanged."""
    while True:
        permutation = rng.permutation(n)
        if not np.any(permutation == np.arange(n)):
            return permutation


def _fork_localization(parsed_info: list, data_types=(), times=(), params=(), ne=0):  # noqa: ANN001, ANN202
    from pipt.misc_tools.cov_regularization import localization

    return localization(parsed_info, list(times), list(data_types), list(params), ne)


def _gies_localized_case(
    rng: np.random.Generator, taper: str, *, prior_only: bool
) -> dict[str, np.ndarray]:
    """RLM-MAC with PET-fork's adaptive localization (per iteration, or ``ini_only``).

    Same set-up as ``_gies_rlmmac_case`` (injected ``z``, no truncation) with
    ``n_data > n_ensemble``, plus an injected derangement for the shuffled-ensemble noise
    estimate. The per-iteration variant goes through ``rlmmac_update.update``'s
    ``autoadaloc`` branch; ``ini_only`` builds the taper as ``assimilation.py`` does and
    then takes the ``ini_only`` branch. The taper is also stored (``out_taper``).
    """
    import pipt.misc_tools.analysis_tools as at
    from pipt.geostat.decomp import Cholesky
    from pipt.update_schemes.gies.rlmmac_update import rlmmac_update

    n_state, n_data, n_ensemble, nstd, damping = 30, 60, 25, 1.0, 0.5
    # ini_only divides the anomalies by the prior std and never scales back (a PET bug);
    # keep the std non-unit there so the parity test can pin that behaviour.
    state_std = rng.uniform(0.5, 2.0, n_state)
    state = rng.standard_normal((n_state, n_ensemble)) * state_std[:, None]
    forward = rng.standard_normal((n_data, n_state)) / state_std[None, :]
    predicted = np.tanh(forward @ state)
    reference_prediction = np.tanh(forward @ state.mean(axis=1))
    observations = rng.standard_normal(n_data)
    variances = rng.uniform(0.05, 0.5, n_data)
    z = rng.standard_normal((n_data, n_ensemble))
    permutation = _derangement(rng, n_ensemble)

    with _global_normals(z):
        perturbed, scale_data = Cholesky().gen_real(
            observations, variances, n_ensemble, return_chol=True
        )
    aug_pred_data = np.hstack([predicted, reference_prediction[:, None]])
    parsed = [["field", [1, 5, 6]], ["autoadaloc", nstd], ["type", taper]]  # prod(field) = n_state
    local = _fork_localization(parsed, ne=n_ensemble)
    scheme = SimpleNamespace(
        pert_preddata=(scale_data ** (-1))[:, None]
        * (aug_pred_data[:, :n_ensemble] - aug_pred_data[:, n_ensemble, None]),
        current_state={"x": state},
        list_states=["x"],
        cell_index=None,
        trunc_energy=1.0,
        keys_da={"localization": parsed + ([["ini_only", "yes"]] if prior_only else [])},
        localization=local,
        prior_info={"x": {}},
        obs_data=[],
        assim_index=["t", []],
        list_datatypes=[],
        state_scaling=state_std,
        proj=(np.eye(n_ensemble) - np.ones((n_ensemble, n_ensemble)) / n_ensemble)
        / np.sqrt(n_ensemble - 1),
        scale_data=scale_data,
        real_obs_data=perturbed,
        aug_pred_data=aug_pred_data,
        ne=n_ensemble,
        lam=damping,
    )
    if prior_only:
        # assimilation.py: taper from the raw prior state and its (member) forecast.
        with _injected_permutation(permutation):
            _, taper_matrix = local.auto_ada_loc(
                state, predicted, ["x"], prior_info={"x": {}}, not_cal_step=True
            )
        scheme.tapering_mtx = taper_matrix
        rlmmac_update.update(scheme)
    else:
        with _injected_permutation(permutation):
            rlmmac_update.update(scheme)
        # The taper rlmmac_update used: recompute it on the same inputs and shuffle.
        pert_state = state @ scheme.proj
        u, s, vt = np.linalg.svd(scheme.pert_preddata, full_matrices=False)
        alpha = damping * np.sum(s**2) / len(s)
        gain = vt.T @ np.diag(s / (s**2 + alpha)) @ u.T
        with _injected_permutation(permutation):
            _, taper_matrix = local.auto_ada_loc(
                pert_state,
                gain @ ((scale_data ** (-1))[:, None] * (perturbed - predicted)),
                ["x"],
                prior_info={"x": {}},
            )
    return {
        "in_state": state,
        "in_state_std": state_std,
        "in_predicted": predicted,
        "in_reference_prediction": reference_prediction,
        "in_observations": observations,
        "in_variances": variances,
        "in_z": z,
        "in_damping": np.array(damping),
        "in_threshold_scale": np.array(nstd),
        "in_permutation": permutation,
        "out_perturbed_observations": perturbed,
        "out_taper": np.asarray(taper_matrix, dtype=np.float64),
        "out_posterior": state + scheme.step,
        "out_misfit": at.calc_objectivefun(perturbed, aug_pred_data, variances),
    }


def _esmda_localized_case(rng: np.random.Generator, taper: str) -> dict[str, np.ndarray]:
    """One ES-MDA step (PET-fork ``approx_update``) with per-iteration adaptive localization.

    PET-fork's ES-MDA (``esmda.calc_analysis``) perturbs the observations with the
    inflated covariance ``alpha C_d`` (injected ``z``), whitens the predicted anomalies with
    that factor and calls ``approx_update.update`` at ``lam = 0``, whose ``autoadaloc``
    branch gives ``auto_ada_loc`` the physical state anomalies and ``X L^-1 (D - Y)``. The
    shuffle of the noise estimate is an injected derangement; no truncation. The taper is
    recomputed on the same inputs and stored (``out_taper``).
    """
    from pipt.geostat.decomp import Cholesky
    from pipt.update_schemes.update_methods_ns.approx_update import approx_update

    n_state, n_data, n_ensemble, nstd, alpha = 30, 60, 25, 1.0, 4.0
    state_std = rng.uniform(0.5, 2.0, n_state)
    state = rng.standard_normal((n_state, n_ensemble)) * state_std[:, None]
    forward = rng.standard_normal((n_data, n_state)) / state_std[None, :]
    predicted = np.tanh(forward @ state)
    observations = rng.standard_normal(n_data)
    variances = rng.uniform(0.05, 0.5, n_data)
    z = rng.standard_normal((n_data, n_ensemble))
    permutation = _derangement(rng, n_ensemble)

    with _global_normals(z):
        perturbed, scale_data = Cholesky().gen_real(
            observations, alpha * variances, n_ensemble, return_chol=True
        )
    proj = (np.eye(n_ensemble) - np.ones((n_ensemble, n_ensemble)) / n_ensemble) / np.sqrt(
        n_ensemble - 1
    )
    parsed = [["field", [1, 5, 6]], ["autoadaloc", nstd], ["type", taper]]  # prod(field) = n_state
    local = _fork_localization(parsed, ne=n_ensemble)
    scheme = SimpleNamespace(
        pert_preddata=(scale_data ** (-1))[:, None] * (predicted @ proj),
        current_state={"x": state},
        list_states=["x"],
        cell_index=None,
        trunc_energy=1.0,
        keys_da={"localization": parsed},
        localization=local,
        prior_info={"x": {}},
        obs_data=[],
        assim_index=["t", []],
        list_datatypes=[],
        state_scaling=state_std,
        proj=proj,
        scale_data=scale_data,
        real_obs_data=perturbed,
        aug_pred_data=predicted,
        ne=n_ensemble,
        lam=0,
    )
    with _injected_permutation(permutation):
        approx_update.update(scheme)
    # PET-fork bug: this branch assigns auto_ada_loc's (step, taper) tuple to self.step,
    # so its ES-MDA would add the taper to the state as well (pet_parity.md). Its RLM-MAC
    # branch unpacks the tuple; the intended step is the first element.
    step = scheme.step[0]
    u, s, vt = np.linalg.svd(scheme.pert_preddata, full_matrices=False)
    gain = vt.T @ np.diag(s / (1 + s**2)) @ u.T
    with _injected_permutation(permutation):
        _, taper_matrix = local.auto_ada_loc(
            state @ proj,
            gain @ ((scale_data ** (-1))[:, None] * (perturbed - predicted)),
            ["x"],
            prior_info={"x": {}},
        )
    return {
        "in_state": state,
        "in_predicted": predicted,
        "in_observations": observations,
        "in_variances": variances,
        "in_z": z,
        "in_alpha": np.array(alpha),
        "in_threshold_scale": np.array(nstd),
        "in_permutation": permutation,
        "out_perturbed_observations": perturbed,
        "out_taper": np.asarray(taper_matrix, dtype=np.float64),
        "out_posterior": state + step,
    }


@case("esmda_localized_adaptive_soft", seed=20261006, source="fork")
def esmda_localized_adaptive_soft(rng: np.random.Generator) -> dict[str, np.ndarray]:
    """ES-MDA step with PET-fork's adaptive localization, soft taper."""
    return _esmda_localized_case(rng, "soft")


@case("esmda_localized_adaptive_hard", seed=20261007, source="fork")
def esmda_localized_adaptive_hard(rng: np.random.Generator) -> dict[str, np.ndarray]:
    """ES-MDA step with PET-fork's adaptive localization, hard taper."""
    return _esmda_localized_case(rng, "hard")


@case("gies_localized_adaptive_hard", seed=20261001, source="fork")
def gies_localized_adaptive_hard(rng: np.random.Generator) -> dict[str, np.ndarray]:
    """Per-iteration adaptive localization, hard threshold (see ``_gies_localized_case``)."""
    return _gies_localized_case(rng, "hard", prior_only=False)


@case("gies_localized_adaptive_soft", seed=20261002, source="fork")
def gies_localized_adaptive_soft(rng: np.random.Generator) -> dict[str, np.ndarray]:
    """Per-iteration adaptive localization, soft (Gaspari-Cohn) taper."""
    return _gies_localized_case(rng, "soft", prior_only=False)


@case("gies_localized_adaptive_sigmoid", seed=20261003, source="fork")
def gies_localized_adaptive_sigmoid(rng: np.random.Generator) -> dict[str, np.ndarray]:
    """Per-iteration adaptive localization, sigmoid taper."""
    return _gies_localized_case(rng, "sigm", prior_only=False)


@case("gies_localized_prior_soft", seed=20261004, source="fork")
def gies_localized_prior_soft(rng: np.random.Generator) -> dict[str, np.ndarray]:
    """``ini_only`` adaptive localization (taper from the prior), soft taper."""
    return _gies_localized_case(rng, "soft", prior_only=True)


@case("distance_masks", seed=20261005, source="fork")
def distance_masks(rng: np.random.Generator) -> dict[str, np.ndarray]:
    """PET-fork's distance-based masks on a 2-D, non-square grid.

    ``localize`` returns the (n_state, n_data) mask for one datum per position, built with
    ``gc`` and ``fb`` tapers, with and without anisotropy, at interior, edge and corner
    positions. PET's ``dist_loc`` builds single-layer masks, hence ``nz = 1``. Positions
    are given in PET's own convention (0-based ``x y z``) and stored as ``in_positions``.
    """
    nx, ny, n_ensemble = 12, 8, 30
    positions = np.array(
        [[5, 3, 0], [6, 4, 0], [0, 0, 0], [11, 7, 0], [0, 5, 0], [9, 0, 0], [0, 7, 0], [11, 0, 0]]
    )
    specs = [("gc", 3, (1.0, 0.0)), ("gc", 4, (2.0, 30.0)), ("fb", 5, (1.0, 0.0))]
    # One datum per (taper spec, position). PET's stencil placement crashes for some
    # positions near the edges (a shape mismatch in _repos_mask); those are recorded in
    # out_failed and their masks left as NaN.
    masks = np.full((len(specs), len(positions), nx * ny), np.nan)
    failed = np.zeros((len(specs), len(positions)), dtype=bool)
    for s_index, (taper, radius, (ratio, angle)) in enumerate(specs):
        for p_index, (x, y, z) in enumerate(positions):
            entry = f"{taper} {x} {y} {z} {radius} 0 {ratio} {angle} d0 0 perm"
            try:
                local = _fork_localization(
                    [["field", [1, ny, nx]], [entry]],
                    data_types=["d0"],
                    times=[0.0],
                    params=["perm"],
                    ne=n_ensemble,
                )
                mask = local.localize(
                    ["d0"],
                    [0.0],
                    ["perm"],
                    n_ensemble,
                    {"perm": {"nx": nx, "ny": ny, "nz": 1}},
                    [[1]],
                )
                masks[s_index, p_index] = np.asarray(mask.todense(), dtype=np.float64)[:, 0]
            except ValueError:
                failed[s_index, p_index] = True
    return {
        "in_grid": np.array([nx, ny, 1]),
        "in_positions": positions,
        "in_radius": np.array([spec[1] for spec in specs], dtype=np.float64),
        "in_anisotropy": np.array([spec[2] for spec in specs], dtype=np.float64),
        "in_taper": np.array([spec[0] for spec in specs]),
        "in_n_ensemble": np.array(n_ensemble),
        "out_masks": masks,
        "out_failed": failed,
    }


class QueuedNormals:
    """Stream stand-in returning pre-drawn standard normals in order, one per ``randn`` call."""

    def __init__(self, draws: list[np.ndarray]) -> None:
        self._draws = list(draws)

    def randn(self, *shape: int) -> np.ndarray:
        """Return the next injected array, checking its shape."""
        draw = self._draws.pop(0)
        if draw.shape != shape:
            raise ValueError(f"PET asked for normals of shape {shape}, next is {draw.shape}")
        return draw.copy()


def _prior_field_case(
    rng: np.random.Generator,
    vario: str,
    aniso: tuple[float, float] = (1.0, 1.0),
    angle: tuple[float, float] = (0.0, 0.0),
) -> dict[str, np.ndarray]:
    """PET's prior sampler (``StateLayout.from_prior_info``) on a square 8x8x2 grid.

    Square, so PET's field is stationary; by default also isotropic and unrotated, where
    PET's covariance is what its keywords describe (see apex.algorithms.geostatistics for
    its rotation and cell-order defects). Each layer's normals are injected in order.
    Layers differ in variance and range.
    """
    from misc.structures.layout import StateLayout

    n, nz, ne = 8, 2, 20
    mean = rng.normal(3.0, 0.5, n * n * nz)
    z = [rng.standard_normal((n * n, ne)) for _ in range(nz)]
    info = {
        "mean": mean,
        "variance": [1.0, 0.5],
        "corr_length": [3.0, 5.0],
        "aniso": list(aniso),
        "angle": list(angle),
        "vario": [vario, vario],
        "nx": n,
        "ny": n,
        "nz": nz,
    }
    field, _ = StateLayout.from_prior_info({"permx": info}, ne, rng=QueuedNormals(z), save=False)
    return {
        "in_mean": mean,
        "in_z": np.array(z),
        "in_grid": np.array([n, n, nz]),
        "in_variance": np.array(info["variance"]),
        "in_range": np.array(info["corr_length"]),
        "in_aniso": np.array(info["aniso"]),
        "in_angle": np.array(info["angle"]),
        "out_field": np.asarray(field, dtype=np.float64),
    }


@case("prior_field_sph", seed=20261101)
def prior_field_sph(rng: np.random.Generator) -> dict[str, np.ndarray]:
    """PET's prior sampler with the spherical variogram."""
    return _prior_field_case(rng, "sph")


@case("prior_field_exp", seed=20261102)
def prior_field_exp(rng: np.random.Generator) -> dict[str, np.ndarray]:
    """PET's prior sampler with its 'exp' variogram (exp(-3 u^1.9))."""
    return _prior_field_case(rng, "exp")


@case("prior_field_cub", seed=20261103)
def prior_field_cub(rng: np.random.Generator) -> dict[str, np.ndarray]:
    """PET's prior sampler with the cubic variogram."""
    return _prior_field_case(rng, "cub")


@case("prior_field_aniso", seed=20261104)
def prior_field_aniso(rng: np.random.Generator) -> dict[str, np.ndarray]:
    """PET's prior sampler with anisotropy and angles (what `apex pet-import` reproduces)."""
    return _prior_field_case(rng, "sph", aniso=(2.0, 3.0), angle=(30.0, -20.0))


def _smooth_field(rng: np.random.Generator, grid: tuple[int, ...]) -> np.ndarray:
    """Return a smooth random field on ``grid`` (a few low-frequency modes)."""
    axes = np.meshgrid(*[np.linspace(0.0, 1.0, n) for n in grid], indexing="ij")
    field = np.zeros(grid)
    for _ in range(4):
        freq = rng.uniform(0.5, 3.0, len(grid))
        phase = rng.uniform(0.0, 2 * np.pi)
        field += rng.normal() * np.cos(
            sum(f * a for f, a in zip(freq, axes, strict=True)) * np.pi + phase
        )
    return field


def _wavelet_case(
    rng: np.random.Generator,
    grid: tuple[int, ...],
    *,
    wname: str,
    level: int,
    order: str,
    colored_noise: bool,
    threshold_rule: str = "universal",
    th_mult: float = 1.0,
    use_hard_th: bool = True,
    keep_ca: bool = False,
    masked: bool = False,
    inactive_value: float | None = None,
    min_noise: float = 1e-9,
    ne: int = 5,
) -> dict[str, np.ndarray]:
    """PET's ``SparseRepresentation`` on one observed field and ``ne`` predicted members.

    As PET runs it: ``compress(observed, th_mult)`` fixes the leading indices and the
    noise (``DataReader._wavelet_compression``), then ``compress(member)`` reduces each
    predicted member to them (``forecast._row_transform``). Fields are smooth plus white
    noise. PET flattens the mask in C order (pet_parity.md, D2), so masked cases use
    ``order='C'``. Arrays are stored in APEX layout: predictions ``(n, ne)``.
    """
    from pipt.misc_tools.wavelet_tools import SparseRepresentation

    mask = np.ones(grid, dtype=bool)
    if masked:
        mask = rng.uniform(size=grid) > 0.2  # about 20% inactive cells
    flat_mask = mask.flatten()  # PET's organize_sparse_representation: C order
    truth = _smooth_field(rng, grid)
    # PET's field vector is in `order`; build it from the grid in that order.
    observed = truth.reshape(-1, order=order)[flat_mask] + 0.05 * rng.standard_normal(
        int(mask.sum())
    )
    members = np.column_stack(
        [
            (truth + 0.3 * _smooth_field(rng, grid)).reshape(-1, order=order)[flat_mask]
            + 0.05 * rng.standard_normal(int(mask.sum()))
            for _ in range(ne)
        ]
    )
    options = {
        "dim": list(grid),
        "mask": flat_mask,
        "level": level,
        "wname": wname,
        "colored_noise": colored_noise,
        "threshold_rule": threshold_rule,
        "th_mult": th_mult,
        "use_hard_th": use_hard_th,
        "keep_ca": keep_ca,
        "inactive_value": inactive_value,
        "order": order,
        "min_noise": min_noise,
    }
    rep = SparseRepresentation(options)
    compressed, wdec_rec = rep.compress(observed, th_mult=th_mult)
    reconstruction = rep.reconstruct(wdec_rec)
    predicted = np.column_stack([rep.compress(members[:, j])[0] for j in range(ne)])
    n_ca = int(np.prod(np.asarray(wdec_rec[0]).shape))
    return {
        "in_observed": observed,
        "in_members": members,
        "in_mask": mask,
        "in_grid": np.array(grid),
        "in_level": np.array(level),
        "in_th_mult": np.array(th_mult),
        "in_min_noise": np.array(min_noise),
        "in_inactive_value": np.array(np.nan if inactive_value is None else inactive_value),
        "out_compressed": np.asarray(compressed, dtype=np.float64),
        "out_est_noise": np.asarray(rep.est_noise, dtype=np.float64),
        # Leading indices into APEX's single coefficient vector (approximation first).
        "out_retained": np.concatenate([rep.ca_leading_index, rep.cd_leading_index + n_ca]),
        "out_num_total_coeff": np.array(rep.num_total_coeff),
        "out_inactive_value": np.array(rep.options["inactive_value"]),
        "out_reconstruction": np.asarray(reconstruction, dtype=np.float64),
        "out_predicted": predicted,
    }


@case("wavelet_white_hard_3d", seed=20261201)
def wavelet_white_hard_3d(rng: np.random.Generator) -> dict[str, np.ndarray]:
    """3-D, db2, white noise, universal rule, hard thresholding, Fortran order (PET's test)."""
    return _wavelet_case(rng, (16, 12, 6), wname="db2", level=2, order="F", colored_noise=False)


@case("wavelet_coloured_masked_2d", seed=20261202)
def wavelet_coloured_masked_2d(rng: np.random.Generator) -> dict[str, np.ndarray]:
    """2-D, sym4, coloured noise, universal rule, hard, masked (C order), mean fill."""
    return _wavelet_case(
        rng, (40, 28), wname="sym4", level=3, order="C", colored_noise=True, masked=True
    )


@case("wavelet_bayesian_keepca_3d", seed=20261203)
def wavelet_bayesian_keepca_3d(rng: np.random.Generator) -> dict[str, np.ndarray]:
    """3-D, haar, coloured noise, PET's Bayesian rule, kept approximation, given fill value."""
    return _wavelet_case(
        rng,
        (12, 10, 8),
        wname="haar",
        level=2,
        order="C",
        colored_noise=True,
        threshold_rule="bayesian",
        th_mult=0.5,
        keep_ca=True,
        masked=True,
        inactive_value=0.0,
    )


@case("wavelet_white_soft_2d", seed=20261204)
def wavelet_white_soft_2d(rng: np.random.Generator) -> dict[str, np.ndarray]:
    """2-D, db3, white noise, soft thresholding (PET does not shrink the predictions, D1)."""
    return _wavelet_case(
        rng,
        (32, 24),
        wname="db3",
        level=2,
        order="F",
        colored_noise=False,
        use_hard_th=False,
        th_mult=1.5,
    )


@case("wavelet_odd_soft_coloured_3d", seed=20261205)
def wavelet_odd_soft_coloured_3d(rng: np.random.Generator) -> dict[str, np.ndarray]:
    """Odd 3-D grid (reconstruction crop), db2, coloured noise, soft, kept approximation."""
    return _wavelet_case(
        rng,
        (15, 12, 7),
        wname="db2",
        level=1,
        order="F",
        colored_noise=True,
        use_hard_th=False,
        keep_ca=True,
    )


#: The PET `compress` block of the `pet_import_compress` case (TOML values, PET's spelling).
IMPORT_COMPRESS = {
    "compress_data": ["avo", "ts"],
    "dim": [8, 6, 4],
    "mask": ["mask_0.npz", "mask_1.npz", "mask_2.npz"],
    "level": 1,
    "wname": "db2",
    "threshold_rule": "universal",
    "th_mult": 0.5,
    "order": "F",
    "min_noise": [1e-9, 1e-6, 1e-9],
    "colored_noise": "yes",
    "use_hard_th": "yes",
    "keep_ca": "no",
    "inactive_value": 0.0,
}

#: Row labels in file order (not sorted) and the cells of each row, in column order.
IMPORT_LABELS = [3000, 1000, 4000, 2000]
IMPORT_COLUMNS = ["avo", "ts", "grav"]
IMPORT_FIELDS = {
    (3000, "avo"),
    (3000, "ts"),
    (1000, "avo"),
    (1000, "ts"),
    (2000, "avo"),
    (2000, "ts"),
}


@case("pet_import_compress", seed=20261206)
def pet_import_compress(rng: np.random.Generator) -> dict[str, np.ndarray]:
    """PET's reader on a small seismic case with `compress`: what `apex pet-import` must build.

    Two compressed types, `avo` and `ts` (`.npz` cells), interleaved in one table with a
    scalar `grav`, with rows out of label order. PET numbers vintages with one counter
    walking rows then columns: avo@3000, ts@3000, avo@1000 get masks 0-2 (mask 0 all
    active; masks 1 and 2 3-D with inactive cells, applied in PET's C-order flattening to
    F-ordered data, D2) and their `min_noise` entries; ts@1000 and the 2000 row lie beyond
    the masks and stay uncompressed, so the cutoff falls inside a row. PET builds its data
    and variance vectors (`DataReader` + `DataLayout`); each cell's slice is stored.
    """
    import tempfile

    import pandas as pd
    from misc.read_input_csv import DataReader
    from misc.structures.layout import DataLayout
    from pipt.misc_tools.extract_tools import organize_sparse_representation

    grid = tuple(IMPORT_COMPRESS["dim"])
    n_cells = int(np.prod(grid))
    masks = [np.ones(grid, dtype=bool)] + [rng.uniform(size=grid) > 0.25 for _ in range(2)]
    # PET's walk order, and the flat mask each vintage's data are indexed with.
    walk = [
        (label, c) for label in IMPORT_LABELS for c in IMPORT_COLUMNS if (label, c) in IMPORT_FIELDS
    ]
    fields = {}
    for vintage, cell in enumerate(walk):
        active = masks[vintage].flatten() if vintage < len(masks) else np.ones(n_cells, dtype=bool)
        field = _smooth_field(rng, grid).reshape(-1, order="F") + 0.05 * rng.standard_normal(
            n_cells
        )
        fields[cell] = field[active]
    grav = {label: float(rng.normal(5.0, 1.0)) for label in IMPORT_LABELS}

    def data_cell(label: int, column: str) -> str:
        if column == "grav":
            return repr(grav[label])
        return f"{column}_{label}.npz" if (label, column) in fields else ""

    def var_cell(label: int, column: str) -> str:
        if column == "grav":
            return "['abs', 0.01]"
        return "['abs', 0.0004]" if (label, column) in fields else ""

    header = ",".join(["time", *IMPORT_COLUMNS])
    data_rows = [header] + [
        ",".join([str(label)] + [data_cell(label, c) for c in IMPORT_COLUMNS])
        for label in IMPORT_LABELS
    ]
    var_rows = [header] + [
        ",".join(
            [str(label)]
            + [f'"{var_cell(label, c)}"' if var_cell(label, c) else "" for c in IMPORT_COLUMNS]
        )
        for label in IMPORT_LABELS
    ]
    cwd = Path.cwd()
    with tempfile.TemporaryDirectory() as tmp:
        os.chdir(tmp)
        try:
            for i, mask in enumerate(masks):
                np.savez(f"mask_{i}.npz", mask=mask)
            for (label, column), values in fields.items():
                np.savez(f"{column}_{label}.npz", values)
            # PET reads the tables as pickled object-dtype frames, as its own compression
            # test writes them: with pandas >= 3 it cannot store a loaded field into the
            # string column read_csv makes of `.npz` cells, and read_csv's parse_dates
            # turns integer labels into dates (pet_parity.md). The CSVs stored below (what
            # `apex pet-import` reads) hold the same cells.
            data = pd.DataFrame(index=IMPORT_LABELS, columns=IMPORT_COLUMNS, dtype=object)
            var = pd.DataFrame(index=IMPORT_LABELS, columns=IMPORT_COLUMNS, dtype=object)
            data.index.name = var.index.name = "time"
            for label in IMPORT_LABELS:
                for column in IMPORT_COLUMNS:
                    cell = data_cell(label, column)
                    data.at[label, column] = grav[label] if column == "grav" else (cell or np.nan)
                    var.at[label, column] = var_cell(label, column) or None
            data.to_pickle("data.pkl")
            var.to_pickle("var.pkl")
            sparse = organize_sparse_representation(dict(IMPORT_COMPRESS))
            reader = DataReader({"data": "data.pkl", "datavar": "var.pkl"}, sparse_info=sparse)
            frame = reader.get_data()
            variance = reader.get_variance(frame, reader.sparse_data)
            layout = DataLayout.from_frame(frame)
            values, variances = layout.vector(frame), layout.vector(variance)
        finally:
            os.chdir(cwd)
    out = {}
    for label in IMPORT_LABELS:
        for column in IMPORT_COLUMNS:
            if column != "grav" and (label, column) not in fields:
                continue
            rows = layout.row(label, column).rows
            out[f"out_{column}_{label}"] = values[rows]
            out[f"out_{column}_var_{label}"] = variances[rows]
    return {
        **{f"in_{column}_{label}": v for (label, column), v in fields.items()},
        **{f"in_mask_{i}": m for i, m in enumerate(masks)},
        "in_data_csv": np.array("\n".join(data_rows) + "\n"),
        "in_var_csv": np.array("\n".join(var_rows) + "\n"),
        **out,
    }


@case("metrics_rmse_ssim", seed=20261301, source="fork")
def metrics_rmse_ssim(rng: np.random.Generator) -> dict[str, np.ndarray]:
    """PET-fork's model RMSE and SSIM (``Plotting/metric_analyses.py``), as in its Brugge study.

    A 10x8x9 grid with about 15% inactive cells (a 4x2x3 interior of 7-cell windows), two
    parameters (``permx``, ``poro``) and two ensembles of 6 members (the "initial" one and
    one analysis step). They are written in the files PET-fork's functions read: the
    truth with upper-case keys, ``init_ens.npz``, ``debug_analysis_step_1.npz`` with a
    pickled ``state`` dict, and ``actnum.npz``. Results are ``(n_steps, n_members)`` per
    group.
    """
    import tempfile

    from Plotting.metric_analyses import calculate_rmse_and_model_change, calculate_ssim

    grid = [10, 8, 9]
    n_cells, ne = int(np.prod(grid)), 6
    actnum = (rng.uniform(size=n_cells) > 0.15).astype(int)
    truth = {
        "permx": 5.0 + _smooth_field(rng, tuple(grid)).reshape(-1, order="F"),
        "poro": 0.2 + 0.05 * _smooth_field(rng, tuple(grid)).reshape(-1, order="F"),
    }
    ensembles = []
    for spread in (1.0, 0.4):
        ensembles.append(
            {
                name: truth[name][:, None] + spread * scale * rng.standard_normal((n_cells, ne))
                for name, scale in (("permx", 1.0), ("poro", 0.05))
            }
        )
    cwd = Path.cwd()
    with tempfile.TemporaryDirectory() as tmp:
        os.chdir(tmp)
        try:
            np.savez("actnum.npz", actnum=actnum)
            np.savez("truth.npz", PERMX=truth["permx"], PORO=truth["poro"])
            np.savez("init_ens.npz", **ensembles[0])
            np.savez("debug_analysis_step_1.npz", state=np.array(ensembles[1], dtype=object))
            rmse, _ = calculate_rmse_and_model_change(
                1,
                "truth.npz",
                ["PERMX", "PORO"],
                init_ens_file="init_ens.npz",
                ens_file_pattern="debug_analysis_step_",
                actnum="actnum.npz",
            )
            mssim = calculate_ssim(
                1,
                "truth.npz",
                ["PERMX", "PORO"],
                list(grid),
                init_ens_file="init_ens.npz",
                ens_file_pattern="debug_analysis_step_",
                actnum="actnum.npz",
            )
        finally:
            os.chdir(cwd)
    return {
        "in_grid": np.array(grid),
        "in_actnum": actnum,
        **{f"in_truth_{name}": v for name, v in truth.items()},
        **{f"in_ens{k}_{name}": e[name] for k, e in enumerate(ensembles) for name in e},
        **{f"out_rmse_{label}": np.asarray(v, dtype=np.float64) for label, v in rmse.items()},
        **{f"out_ssim_{label}": np.asarray(v, dtype=np.float64) for label, v in mssim.items()},
    }


@case("metrics_model_change", seed=20261302, source="fork")
def metrics_model_change(rng: np.random.Generator) -> dict[str, np.ndarray]:
    """PET-fork's model change between iterations (``calculate_rmse_and_model_change``).

    As ``metrics_rmse_ssim`` (the same files, a masked grid, two parameters), with three
    ensembles of 6 members (the initial one and two analysis steps), so that both steps'
    changes are computed against the previous ensemble. Results are ``(n_steps, n_members)``
    per group, for steps 1 and 2.
    """
    import tempfile

    from Plotting.metric_analyses import calculate_rmse_and_model_change

    grid = [6, 5, 4]
    n_cells, ne = int(np.prod(grid)), 6
    actnum = (rng.uniform(size=n_cells) > 0.2).astype(int)
    truth = {
        "permx": 5.0 + rng.standard_normal(n_cells),
        "poro": 0.2 + 0.02 * rng.standard_normal(n_cells),
    }
    ensembles = [
        {
            name: truth[name][:, None] + spread * scale * rng.standard_normal((n_cells, ne))
            for name, scale in (("permx", 1.0), ("poro", 0.05))
        }
        for spread in (1.0, 0.6, 0.3)
    ]
    cwd = Path.cwd()
    with tempfile.TemporaryDirectory() as tmp:
        os.chdir(tmp)
        try:
            np.savez("actnum.npz", actnum=actnum)
            np.savez("truth.npz", PERMX=truth["permx"], PORO=truth["poro"])
            np.savez("init_ens.npz", **ensembles[0])
            for k in (1, 2):
                np.savez(f"debug_analysis_step_{k}.npz", state=np.array(ensembles[k], dtype=object))
            _, change = calculate_rmse_and_model_change(
                2,
                "truth.npz",
                ["PERMX", "PORO"],
                init_ens_file="init_ens.npz",
                ens_file_pattern="debug_analysis_step_",
                actnum="actnum.npz",
            )
        finally:
            os.chdir(cwd)
    return {
        "in_actnum": actnum,
        **{f"in_ens{k}_{name}": e[name] for k, e in enumerate(ensembles) for name in e},
        **{f"out_change_{label}": np.asarray(v, dtype=np.float64) for label, v in change.items()},
    }


# --------------------------------------------------------------------------------------------
# popt: ensemble optimization
# --------------------------------------------------------------------------------------------


class RecordingRandom:
    """PET's ensemble ``rng`` that records every draw (a ``RandomState`` underneath).

    The parity tests replay the recorded draws in order, so a whole PET optimization
    (backtracking, resampling, SMC resampling) can be compared with APEX's. Draws are stored
    as PET returns them (one draw per row): that is the draw source's layout in APEX too.
    """

    def __init__(self, seed: int) -> None:
        self.state = np.random.RandomState(seed)
        self.draws: list[tuple[str, np.ndarray]] = []
        # What each draw was asked for, so the tests check APEX asks for the same.
        self.arguments: list[dict[str, np.ndarray]] = []

    def multivariate_normal(self, mean, cov, size=None, **kwargs):  # noqa: ANN001, ANN003, ANN201
        """Draw as ``RandomState.multivariate_normal`` and record the draw."""
        out = self.state.multivariate_normal(mean, cov, size)
        self.draws.append(("mvn", np.array(out, dtype=float)))
        self.arguments.append({"mean": np.array(mean, float), "cov": np.array(cov, float)})
        return out

    def choice(self, a, size=None, replace=True, p=None):  # noqa: ANN001, ANN201
        """Draw as ``RandomState.choice`` and record the draw."""
        out = self.state.choice(a, size, replace=replace, p=p)
        self.draws.append(("choice", np.array(out, dtype=np.int64)))
        self.arguments.append({"p": np.array(p, float)})
        return out

    def arrays(self) -> dict[str, np.ndarray]:
        """Return the draws as ``in_draw_<k>_<kind>`` arrays."""
        out = {f"in_draw_{k:03d}_{kind}": value for k, (kind, value) in enumerate(self.draws)}
        for k, arguments in enumerate(self.arguments):
            out.update({f"in_drawarg_{k:03d}_{name}": v for name, v in arguments.items()})
        return out


@contextmanager
def _in_temporary_folder() -> Iterator[None]:
    # PET's ensembles write prior files into the working folder.
    import tempfile

    previous = Path.cwd()
    with tempfile.TemporaryDirectory(prefix="pet_popt_") as folder:
        os.chdir(folder)
        try:
            yield
        finally:
            os.chdir(previous)


def _rosen_objective(x: np.ndarray, *args: object, **kwargs: object) -> np.ndarray:
    from scipy.optimize import rosen

    return np.atleast_1d(rosen(x))


def _quadratic_objective(x: np.ndarray, *args: object, **kwargs: object) -> np.ndarray:
    # PET's quadratic is 0.5 ||x - 1||^2 per column (A = I / 2, b = 1); its own function
    # reads `r` from a keyword the optimizers never pass, so the penalty is added here.
    x2 = x if x.ndim == 2 else x[:, None]
    value = 0.5 * np.sum((x2 - 1.0) ** 2, axis=0)
    epf = kwargs.get("epf")
    if isinstance(epf, dict) and epf:
        from popt.cost_functions.epf import epf as penalty_of

        c_eq = np.sum(x2, axis=0, keepdims=True) - (x2.shape[0] + 1)
        c_iq = -x2[:1]
        penalty = penalty_of(epf["r"], c_eq=c_eq, c_iq=c_iq)
        epf["penalty"] = penalty
        value = value + penalty
    return value


_OBJECTIVES = {"rosenbrock": _rosen_objective, "quadratic": _quadratic_objective}


def _trajectory(optimizer: object, history: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    """Stack the per-iterate records and the final counters of a PET optimizer."""
    out = {f"out_{key}": np.array([row[key] for row in history]) for key in history[0]}
    out["out_message"] = np.array(optimizer.conv_msg)  # type: ignore[attr-defined]
    out["out_final_x"] = np.array(optimizer.optimize_results["x"], dtype=float)  # type: ignore[attr-defined]
    out["out_final_nfev"] = np.array(optimizer.fun.nfev)  # type: ignore[attr-defined]
    jac = optimizer.jac  # type: ignore[attr-defined]
    out["out_final_njev"] = np.array(jac.nfev if jac else 0)
    out["out_final_iteration"] = np.array(optimizer.iteration)  # type: ignore[attr-defined]
    return out


def _recorder(history: list[dict[str, np.ndarray]]) -> Callable[[object], None]:
    def callback(optimizer: object) -> None:
        history.append(
            {
                "xk": np.array(optimizer.xk, dtype=float),  # type: ignore[attr-defined]
                "fk": np.atleast_1d(np.mean(optimizer.fk)).astype(float),  # type: ignore[attr-defined]
                "nfev": np.array(optimizer.fun.nfev),  # type: ignore[attr-defined]
                "iteration": np.array(optimizer.iteration),  # type: ignore[attr-defined]
            }
        )

    return callback


def _gaussian_enopt_case(
    rng: np.random.Generator,
    objective: str,
    x0: list[float],
    variance: float,
    ne: int,
    options: dict[str, object],
    *,
    natural_gradient: bool = True,
    with_hessian: bool = False,
    optimizer_class: str = "EnOpt",
) -> dict[str, np.ndarray]:
    """Run PET's EnOpt (or SmcOpt) with a recorded RandomState; return inputs and the path."""
    from popt.ensembles import GaussianEnsemble
    from popt.optimization_methods import EnOpt, SmcOpt

    seed = int(rng.integers(2**31 - 1))
    config = {
        "ne": ne,
        "natural_gradient": natural_gradient,
        "controls": {"x": {"mean": x0, "var": variance, "limits": [-2, 2]}},
    }
    history: list[dict[str, np.ndarray]] = []
    with _in_temporary_folder():
        ensemble = GaussianEnsemble(config, None, _OBJECTIVES[objective])
        recorder = RecordingRandom(seed)
        ensemble.rng = recorder
        # PET's ensemble overwrites its covariance with every gradient's: keep the initial one.
        cov0 = np.array(ensemble.get_cov(), dtype=float)
        common = {"bounds": ensemble.get_bounds(), "logit": False, "callback": _recorder(history)}
        if optimizer_class == "SmcOpt":
            optimizer = SmcOpt(
                ensemble.get_state(),
                ensemble.function,
                sens=ensemble.calc_ensemble_weights,
                args=(ensemble.get_cov(),),
                **common,
                **options,
            )
        else:
            optimizer = EnOpt(
                ensemble.get_state(),
                ensemble.function,
                jac=ensemble.gradient,
                hess=ensemble.hessian if with_hessian else None,
                args=(ensemble.get_cov(),),
                **common,
                **options,
            )
        optimizer.run_optimization()
    return {
        "in_x0": np.array(x0, dtype=float),
        "in_cov": cov0,
        "in_lower": np.full(len(x0), -2.0),
        "in_upper": np.full(len(x0), 2.0),
        **recorder.arrays(),
        **_trajectory(optimizer, history),
    }


@case("popt_enopt_gd", seed=20261401)
def popt_enopt_gd(rng: np.random.Generator) -> dict[str, np.ndarray]:
    """EnOpt, GD, natural gradient, on Rosenbrock (3 controls, 10 members, 15 iterations)."""
    return _gaussian_enopt_case(rng, "rosenbrock", [0.5, 0.5, 0.5], 0.01, 10, {"maxiter": 15})


@case("popt_enopt_momentum", seed=20261402)
def popt_enopt_momentum(rng: np.random.Generator) -> dict[str, np.ndarray]:
    """EnOpt, GD with momentum and Nesterov, averaged gradient, resampling, on Rosenbrock."""
    options = {"maxiter": 15, "beta": 0.5, "nesterov": True, "resample": 2, "alpha": 0.2}
    return _gaussian_enopt_case(
        rng, "rosenbrock", [0.5, -0.5, 0.5], 0.01, 12, options, natural_gradient=False
    )


@case("popt_enopt_adam", seed=20261403)
def popt_enopt_adam(rng: np.random.Generator) -> dict[str, np.ndarray]:
    """EnOpt with Adam (PET passes beta as Adam's beta1), on Rosenbrock."""
    options = {"maxiter": 15, "optimizer": "Adam", "beta": 0.9, "alpha": 0.05}
    return _gaussian_enopt_case(rng, "rosenbrock", [0.5, 0.5, 0.5], 0.01, 10, options)


@case("popt_enopt_adamax", seed=20261404)
def popt_enopt_adamax(rng: np.random.Generator) -> dict[str, np.ndarray]:
    """EnOpt with AdaMax (no normalisation), on Rosenbrock."""
    options = {"maxiter": 15, "optimizer": "AdaMax", "beta": 0.9, "alpha": 0.05}
    return _gaussian_enopt_case(rng, "rosenbrock", [0.5, 0.5, 0.5], 0.01, 10, options)


@case("popt_enopt_hessian", seed=20261405)
def popt_enopt_hessian(rng: np.random.Generator) -> dict[str, np.ndarray]:
    """EnOpt with the ensemble Hessian in the direction and the covariance update (quadratic)."""
    options = {"maxiter": 10, "hessian": True, "alpha": 0.5, "alpha_cov": 0.01, "resample": 2}
    return _gaussian_enopt_case(
        rng, "quadratic", [-1.0, 0.0, 0.5, 1.5], 0.04, 60, options, with_hessian=True
    )


@case("popt_enopt_steihaug", seed=20261406)
def popt_enopt_steihaug(rng: np.random.Generator) -> dict[str, np.ndarray]:
    """EnOpt with the CG-Steihaug step and the covariance update, on the quadratic.

    60 members keep the ensemble Hessian positive definite along every CG direction: at
    negative curvature PET's Steihaug computes the boundary step from the residual instead
    of the iterate (a PET bug APEX does not copy; unit-tested in APEX).
    """
    options = {"maxiter": 10, "optimizer": "Steihaug", "alpha_cov": 0.01}
    return _gaussian_enopt_case(
        rng, "quadratic", [-1.0, 0.0, 0.5, 1.5], 0.04, 60, options, with_hessian=True
    )


@case("popt_enopt_transform", seed=20261407)
def popt_enopt_transform(rng: np.random.Generator) -> dict[str, np.ndarray]:
    """EnOpt in the unit cube of the bounds, with the Hessian's covariance update (Rosenbrock)."""
    options = {"maxiter": 12, "transform": True, "alpha_cov": 0.01}
    return _gaussian_enopt_case(
        rng, "rosenbrock", [0.5, 0.5, 0.5], 0.01, 10, options, with_hessian=True
    )


@case("popt_enopt_epf", seed=20261408)
def popt_enopt_epf(rng: np.random.Generator) -> dict[str, np.ndarray]:
    """EnOpt with the external penalty loop on the constrained quadratic."""
    options = {
        "maxiter": 8,
        "alpha": 0.3,
        "epf": {"r": 1.0, "r_factor": 4.0, "conv_crit": 1e-3, "max_epf_iter": 4},
    }
    return _gaussian_enopt_case(rng, "quadratic", [-0.5, 0.5, 0.5], 0.01, 12, options)


@case("popt_smcopt", seed=20261409)
def popt_smcopt(rng: np.random.Generator) -> dict[str, np.ndarray]:
    """SmcOpt with half the particles surviving and resampling, on Rosenbrock."""
    options = {"maxiter": 12, "survival_factor": 0.5, "resample": 1, "alpha": 0.3}
    return _gaussian_enopt_case(
        rng, "rosenbrock", [0.5, 0.5, 0.5], 0.01, 20, options, optimizer_class="SmcOpt"
    )


def _genopt_case(
    rng: np.random.Generator, marginal: str, options: dict[str, object], *, cma: bool
) -> dict[str, np.ndarray]:
    """Run PET's GenOpt with a recorded RandomState on Rosenbrock (bounds [-2, 2])."""
    from popt.ensembles import GeneralizedEnsemble
    from popt.optimization_methods import GenOpt
    from popt.optimization_methods.subroutines.cma import CMA as PetCMA

    seed = int(rng.integers(2**31 - 1))
    x0 = [0.5, 0.5, 0.5]
    config = {
        "ne": 16,
        "marginal": marginal,
        "controls": {"x": {"mean": x0, "var": 0.01, "limits": [-2, 2]}},
    }
    history: list[dict[str, np.ndarray]] = []
    with _in_temporary_folder():
        ensemble = GeneralizedEnsemble(config, None, _rosen_objective)
        recorder = RecordingRandom(seed)
        ensemble.rng = recorder
        theta0 = np.array(ensemble.get_theta(), dtype=float)
        cov0 = np.array(ensemble.get_cov(), dtype=float)
        corr_adapt = PetCMA(ne=16, dim=3, corr_update=True) if cma else None
        optimizer = GenOpt(
            ensemble.get_state(),
            ensemble.function,
            jac=ensemble.gradient,
            jac_mut=ensemble.mutation_gradient,
            corr_adapt=corr_adapt,
            args=(ensemble.get_theta(), ensemble.get_corr()),
            bounds=ensemble.get_bounds(),
            logit=False,
            callback=_recorder(history),
            **options,
        )
        optimizer.run_optimization()
    return {
        "in_x0": np.array(x0, dtype=float),
        "in_cov": cov0,
        "in_lower": np.full(3, -2.0),
        "in_upper": np.full(3, 2.0),
        "in_theta0": theta0,
        **recorder.arrays(),
        **_trajectory(optimizer, history),
        "out_theta": np.array(optimizer.theta, dtype=float),
        "out_corr": np.array(optimizer.corr, dtype=float),
    }


@case("popt_genopt_betamc_cma", seed=20261410)
def popt_genopt_betamc_cma(rng: np.random.Generator) -> dict[str, np.ndarray]:
    """GenOpt, BetaMC marginal, theta adapted and the correlation by CMA (two draws/iter)."""
    return _genopt_case(rng, "BetaMC", {"maxiter": 10, "alpha": 0.1, "alpha_theta": 0.1}, cma=True)


@case("popt_genopt_truncgauss", seed=20261411)
def popt_genopt_truncgauss(rng: np.random.Generator) -> dict[str, np.ndarray]:
    """GenOpt, truncated-Gaussian marginal, theta fixed (PET's theta derivative is in mu)."""
    options = {"maxiter": 10, "alpha": 0.1, "alpha_theta": 0.0}
    return _genopt_case(rng, "TruncGaussian", options, cma=False)


def _deterministic_case(
    method_class: str, x0: list[float], options: dict[str, object], *, hess: object = True
) -> dict[str, np.ndarray]:
    from popt.optimization_methods import LineSearch, TrustRegion
    from scipy.optimize import rosen, rosen_der, rosen_hess

    history: list[dict[str, np.ndarray]] = []
    with _in_temporary_folder():
        common = {"logit": False, "callback": _recorder(history)}
        if method_class == "LineSearch":
            optimizer = LineSearch(
                np.array(x0), rosen, jac=rosen_der, hess=rosen_hess, **common, **options
            )
        else:
            optimizer = TrustRegion(
                np.array(x0),
                rosen,
                rosen_der,
                rosen_hess if hess is True else hess,
                **common,
                **options,
            )
        optimizer.run_optimization()
    return {"in_x0": np.array(x0, dtype=float), **_trajectory(optimizer, history)}


@case("popt_linesearch_bfgs", seed=20261412)
def popt_linesearch_bfgs(rng: np.random.Generator) -> dict[str, np.ndarray]:
    """LineSearch, BFGS with the Wolfe line search, on Rosenbrock (4 controls)."""
    options = {"method": "BFGS", "maxiter": 60}
    return _deterministic_case("LineSearch", [-1.2, 1.0, -0.5, 0.8], options)


@case("popt_linesearch_newton_cg", seed=20261413)
def popt_linesearch_newton_cg(rng: np.random.Generator) -> dict[str, np.ndarray]:
    """LineSearch, Newton-CG with the Wolfe line search, on Rosenbrock."""
    options = {"method": "Newton-CG", "maxiter": 60}
    return _deterministic_case("LineSearch", [-1.2, 1.0, -0.5, 0.8], options)


@case("popt_linesearch_gd_backtracking", seed=20261414)
def popt_linesearch_gd_backtracking(rng: np.random.Generator) -> dict[str, np.ndarray]:
    """LineSearch, steepest descent with backtracking and gradient-ratio step adaptation."""
    options = {"method": "GD", "lsmethod": 0, "step_size_adapt": 2, "maxiter": 40}
    return _deterministic_case("LineSearch", [-1.2, 1.0, -0.5, 0.8], options)


@case("popt_trust_region_iterative", seed=20261415)
def popt_trust_region_iterative(rng: np.random.Generator) -> dict[str, np.ndarray]:
    """TrustRegion with the exact (iterative) subproblem, on Rosenbrock."""
    return _deterministic_case("TrustRegion", [-1.2, 1.0, -0.5, 0.8], {"maxiter": 60})


@case("popt_trust_region_bfgs_steihaug", seed=20261416)
def popt_trust_region_bfgs_steihaug(rng: np.random.Generator) -> dict[str, np.ndarray]:
    """TrustRegion with BFGS and the CG-Steihaug subproblem, on Rosenbrock."""
    options = {"maxiter": 80, "method": "CG-Steihaug"}
    return _deterministic_case("TrustRegion", [-1.2, 1.0, -0.5, 0.8], options, hess="BFGS")


@case("popt_generalized_gradients", seed=20261417)
def popt_generalized_gradients(rng: np.random.Generator) -> dict[str, np.ndarray]:
    """Compute the generalized ensemble's gradient, Hessian and mutation gradient per marginal.

    PET draws the copula's normals through the recorded RandomState; its samples (members in
    rows) are stored transposed, ensemble last. TruncGaussian's mutation gradient is not
    stored (PET's is the derivative in the mean).
    """
    from popt.ensembles import GeneralizedEnsemble

    x0 = [0.3, 0.6, 0.45]
    out: dict[str, np.ndarray] = {"in_x0": np.array(x0)}
    for marginal in ("BetaMC", "Beta", "Logistic", "TruncGaussian", "Gaussian"):
        limits = [0, 1] if marginal == "Beta" else [-2, 2]
        config = {
            "ne": 25,
            "marginal": marginal,
            "controls": {"x": {"mean": x0, "var": 0.02, "limits": limits}},
        }
        with _in_temporary_folder():
            ensemble = GeneralizedEnsemble(config, None, _rosen_objective)
            recorder = RecordingRandom(int(rng.integers(2**31 - 1)))
            ensemble.rng = recorder
            x = np.array(x0)
            ensemble.function(x)
            theta, corr = ensemble.get_theta(), ensemble.get_corr()
            gradient = ensemble.gradient(x, theta, corr)
            hessian = ensemble.hessian(x)
            out[f"in_{marginal}_z"] = recorder.draws[0][1]
            out[f"in_{marginal}_theta"] = np.array(theta, dtype=float)
            out[f"out_{marginal}_members"] = np.array(ensemble.enX, dtype=float).T
            out[f"out_{marginal}_values"] = np.array(ensemble.enF, dtype=float)
            out[f"out_{marginal}_gradient"] = np.array(gradient, dtype=float)
            out[f"out_{marginal}_hessian"] = np.array(hessian, dtype=float)
            if marginal == "BetaMC":
                nat = ensemble.mutation_gradient(
                    x, theta, corr, enZ=ensemble.enZ, enX=ensemble.enX, enF=ensemble.enF
                )
                out["out_BetaMC_mutation_gradient"] = np.array(nat, dtype=float)
                out["out_BetaMC_mutation_hessian"] = np.array(ensemble.mutation_hessian(x))
    return out


@case("popt_cma", seed=20261418)
def popt_cma(rng: np.random.Generator) -> dict[str, np.ndarray]:
    """Two CMA updates (equal and log weights; covariance and correlation), random inputs.

    Samples are drawn ensemble last and handed to PET transposed (members in rows).
    """
    from popt.optimization_methods.subroutines.cma import CMA as PetCMA

    ne, dim = 12, 4
    out: dict[str, np.ndarray] = {}
    for label, equal, corr in (("equal_cov", True, False), ("log_corr", False, True)):
        cma = PetCMA(ne, dim, corr_update=corr, equal_weights=equal)
        a = rng.standard_normal((dim, dim))
        cov = a @ a.T / dim + np.eye(dim)
        if corr:
            std = np.sqrt(np.diag(cov))
            cov = cov / np.outer(std, std)
        out[f"in_{label}_cov"] = cov
        for step in range(2):
            samples = rng.standard_normal((dim, ne))
            values = rng.standard_normal(ne)
            direction = rng.standard_normal(dim)
            out[f"in_{label}_samples_{step}"] = samples
            out[f"in_{label}_values_{step}"] = values
            out[f"in_{label}_step_{step}"] = direction
            cov = cma(cov=cov, step=direction, X=samples.T, J=values)
            out[f"out_{label}_cov_{step}"] = np.array(cov)
    return out


#: The PET commit of the NPV objective (removed upstream before e65fb10).
NPV_COMMIT = "fd17979"


@case("popt_npv", seed=20261419)
def popt_npv(rng: np.random.Generator) -> dict[str, np.ndarray]:
    """PET's ``npv`` (``popt/cost_functions/npv.py`` at fd17979) on seeded field totals.

    The function is gone from the reference commit, so its source is read from a PET git
    checkout (``PET_GIT_CHECKOUT``, e.g. the local one of CLAUDE.local.md) at ``NPV_COMMIT``
    and run here; its blob hash is stored. Totals are cumulative and increasing; PET's first
    report point is the start, where they are zero (as in its tutorial). APEX's keys are
    the four totals at the later reports, vector by vector.
    """
    import datetime as dt

    checkout = os.environ.get("PET_GIT_CHECKOUT")
    if not checkout:
        raise SystemExit("set PET_GIT_CHECKOUT to a PET git checkout that has " + NPV_COMMIT)
    spec = f"{NPV_COMMIT}:popt/cost_functions/npv.py"
    source = subprocess.run(
        ["git", "-C", checkout, "show", spec], capture_output=True, text=True, check=True
    ).stdout
    blob = subprocess.run(
        ["git", "-C", checkout, "rev-parse", spec], capture_output=True, text=True, check=True
    ).stdout.strip()
    namespace: dict[str, object] = {}
    exec(compile(source, spec, "exec"), namespace)
    npv = namespace["npv"]

    n_times, ne = 6, 7
    start = dt.datetime(1994, 2, 9)
    dates = [start] + [dt.datetime(1995 + k, 1, 1) for k in range(n_times - 1)]
    vectors = ("fopt", "fgpt", "fwpt", "fwit")
    totals = {v: np.cumsum(rng.uniform(0.0, 1e4, (n_times - 1, ne)), axis=0) for v in vectors}
    pred_data = [{v: np.zeros(ne) for v in vectors}] + [
        {v: totals[v][k] for v in vectors} for k in range(n_times - 1)
    ]
    const = {"wop": 283.05, "wgp": 0.5, "wwp": 37.74, "wwi": 12.58, "disc": 0.08,
             "obj_scaling": -1.0e6}  # fmt: skip
    values = npv(
        pred_data,
        input_dict={"npv_const": list(const.items())},
        true_order=["dates", dates],
    )
    return {
        "in_responses": np.concatenate(
            [totals[v.lower()] for v in ("FOPT", "FGPT", "FWPT", "FWIT")]
        ),
        "in_start": np.array(start.date().isoformat()),
        "in_times": np.array([d.date().isoformat() for d in dates[1:]]),
        "in_prices": np.array(
            [const[k] for k in ("wop", "wgp", "wwp", "wwi", "disc", "obj_scaling")]
        ),
        "in_npv_source": np.array(f"{spec} blob {blob}"),
        "out_values": np.asarray(values, dtype=float),
    }


def _pet_npv_source(module: str) -> tuple[dict[str, object], str]:
    """Exec PET's ``popt/cost_functions/<module>.py`` at ``NPV_COMMIT``; return it and its id.

    The NPV objectives are gone from the reference commit, so their source is read from a
    PET git checkout (``PET_GIT_CHECKOUT``) and run here, as ``popt_npv``.
    """
    checkout = os.environ.get("PET_GIT_CHECKOUT")
    if not checkout:
        raise SystemExit("set PET_GIT_CHECKOUT to a PET git checkout that has " + NPV_COMMIT)
    spec = f"{NPV_COMMIT}:popt/cost_functions/{module}.py"
    source = subprocess.run(
        ["git", "-C", checkout, "show", spec], capture_output=True, text=True, check=True
    ).stdout
    blob = subprocess.run(
        ["git", "-C", checkout, "rev-parse", spec], capture_output=True, text=True, check=True
    ).stdout.strip()
    namespace: dict[str, object] = {"__name__": f"pet_{module}"}
    exec(compile(source, spec, "exec"), namespace)
    return namespace, f"{spec} blob {blob}"


#: The eCalc stand-in of the eCalc NPV cases: CO2 (t/d) linear in the day rates. PET's
#: eCalc (libecalc 8.23) is not installable beside its NumPy 2 environment and its API is
#: gone from libecalc 15, so both sides use this stub; eCalc itself is checked in APEX's own
#: tests against a hand calculation.
ECALC_STUB = {"GAS_PROD": 2.0e-5, "OIL_PROD": 3.0e-3, "WATER_INJ": 1.5e-3, "WIND_POWER": 0.2}


def _stub_rate(row: dict[str, str]) -> float:
    return sum(c * float(row[k]) for k, c in ECALC_STUB.items() if k in row)


@contextmanager
def _stub_ecalc(folder: Path) -> Iterator[None]:
    """Install stand-ins for libecalc, ecalc_cli and pqdm, working in ``folder``.

    The new API (``ecalc_npv``, ``ecalc_pareto_npv``) reads ``ecalc_input.csv``; the old one
    (``ren_npv_co2``) reads the CSV its YAML names. Each returns one component's
    ``co2_fuel_gas`` rate per row: :data:`ECALC_STUB` applied to the row.
    """
    import csv
    import types

    import yaml  # type: ignore[import-untyped]

    def rows(path: Path) -> list[dict[str, str]]:
        with open(path) as handle:
            return [dict(r) for r in csv.DictReader(handle)]

    class Rate:
        def __init__(self, values: list[float]) -> None:
            self.values = np.asarray(values)
            self.unit = "t/d"
            self.timesteps = list(range(len(values)))

    class Graph:
        def __init__(self) -> None:
            self.nodes = {"stub": object()}

        def get_node(self, _: str) -> SimpleNamespace:
            return SimpleNamespace(name="stub")

    class YamlModel:
        def __init__(self, path: Path | None = None, **_: object) -> None:
            self.csv = folder / "ecalc_input.csv"
            if path is not None:  # the old API: the YAML names the CSV
                self.csv = folder / yaml.safe_load(Path(path).read_text())["TIME_SERIES"][0]["FILE"]
            self.graph = Graph()
            self.variables = None

        def get_graph(self) -> Graph:
            return self.graph

    class EnergyCalculator:
        def __init__(self, graph: Graph | None = None, **_: object) -> None:
            self.graph = graph

        def evaluate_energy_usage(self, *_: object) -> None:
            return None

        def evaluate_emissions(self, variables: object, results: object) -> dict[str, object]:
            raise NotImplementedError  # replaced per model below

    modules = {
        name: types.ModuleType(name)
        for name in (
            "libecalc", "libecalc.application", "libecalc.application.energy_calculator",
            "libecalc.common", "libecalc.common.time_utils", "libecalc.presentation",
            "libecalc.presentation.yaml", "libecalc.presentation.yaml.model",
            "libecalc.presentation.yaml.file_configuration_service", "ecalc_cli",
            "ecalc_cli.infrastructure", "ecalc_cli.infrastructure.file_resource_service",
            "pqdm", "pqdm.processes",
        )
    }  # fmt: skip
    models: list[YamlModel] = []

    class Model(YamlModel):
        def __init__(self, *args: object, **kwargs: object) -> None:
            super().__init__(*args, **kwargs)  # type: ignore[arg-type]
            models.append(self)

    class Calculator(EnergyCalculator):
        def evaluate_emissions(self, variables: object, results: object) -> dict[str, object]:
            rates = [_stub_rate(r) for r in rows(models[-1].csv)]
            return {"stub": {"co2_fuel_gas": SimpleNamespace(rate=Rate(rates))}}

    modules["libecalc.application.energy_calculator"].EnergyCalculator = Calculator  # type: ignore[attr-defined]
    modules["libecalc.common.time_utils"].Frequency = SimpleNamespace(NONE=None)  # type: ignore[attr-defined]
    modules["libecalc.presentation.yaml.model"].YamlModel = Model  # type: ignore[attr-defined]
    modules["libecalc.presentation.yaml.file_configuration_service"].FileConfigurationService = (  # type: ignore[attr-defined]
        lambda **_: None
    )
    modules["ecalc_cli.infrastructure.file_resource_service"].FileResourceService = (  # type: ignore[attr-defined]
        lambda **_: None
    )
    modules["pqdm.processes"].pqdm = (  # type: ignore[attr-defined]
        lambda array, function, n_jobs=1, disable=True: [function(a) for a in array]
    )
    saved = {k: sys.modules.get(k) for k in modules}
    here = Path.cwd()
    sys.modules.update(modules)
    os.chdir(folder)
    try:
        yield
    finally:
        os.chdir(here)
        for key, value in saved.items():
            if value is None:
                sys.modules.pop(key, None)
            else:
                sys.modules[key] = value


def _npv_dates(n_times: int) -> list[object]:
    import datetime as dt

    start = dt.datetime(1994, 2, 9)
    return [start] + [dt.datetime(1995 + k, 1, 1) for k in range(n_times - 1)]


@case("popt_ren_npv", seed=20261420)
def popt_ren_npv(rng: np.random.Generator) -> dict[str, np.ndarray]:
    """PET's ``ren_npv`` (fd17979): water injection priced per well, renewable or not.

    Injectors ``inj1``, ``inj2`` (``wwi``) and ``ren1`` (``wrenwi``; PET finds it by
    ``'ren'`` in its ``wwit`` key). Totals as in ``popt_npv``.
    """
    namespace, source = _pet_npv_source("ren_npv")
    ren_npv = namespace["ren_npv"]
    n_times, ne = 6, 7
    dates = _npv_dates(n_times)
    vectors = ("fopt", "fgpt", "fwpt", "wwit:inj1", "wwit:inj2", "wwit:ren1")
    totals = {v: np.cumsum(rng.uniform(0.0, 1e4, (n_times - 1, ne)), axis=0) for v in vectors}
    pred_data = [{v: np.zeros(ne) for v in vectors}] + [
        {v: totals[v][k] for v in vectors} for k in range(n_times - 1)
    ]
    const = {"wop": 283.05, "wgp": 0.5, "wwp": 37.74, "wwi": 12.58, "wrenwi": 4.2,
             "disc": 0.08, "obj_scaling": -1.0e6}  # fmt: skip
    keys_opt = {"npv_const": list(const.items()), "datatype": list(vectors)}
    values = ren_npv(pred_data, {"input_dict": keys_opt, "true_order": ["dates", dates]})  # type: ignore[operator]
    return {
        "in_responses": np.concatenate([totals[v] for v in vectors]),
        "in_vectors": np.array(vectors),
        "in_start": np.array(dates[0].date().isoformat()),  # type: ignore[attr-defined]
        "in_times": np.array([d.date().isoformat() for d in dates[1:]]),  # type: ignore[attr-defined]
        "in_prices": np.array([const[k] for k in const]),
        "in_price_names": np.array(list(const)),
        "in_source": np.array(source),
        "out_values": np.asarray(values, dtype=float),
    }


def _ecalc_totals(rng: np.random.Generator, n_times: int, ne: int) -> dict[str, np.ndarray]:
    return {v: np.cumsum(rng.uniform(1.0, 1e4, (n_times - 1, ne)), axis=0)
            for v in ("fopt", "fgpt", "fwpt", "fwit")}  # fmt: skip


def _ecalc_case(
    rng: np.random.Generator, module: str, const: dict[str, float]
) -> dict[str, np.ndarray]:
    import tempfile

    n_times, ne = 6, 5
    dates = _npv_dates(n_times)
    totals = _ecalc_totals(rng, n_times, ne)
    pred_data = [{v: np.zeros(ne) for v in totals}] + [
        {v: totals[v][k] for v in totals} for k in range(n_times - 1)
    ]
    keys_opt = {"npv_const": list(const.items())}
    with tempfile.TemporaryDirectory() as folder, _stub_ecalc(Path(folder)):
        namespace, source = _pet_npv_source(module)
        function = namespace[module]
        kwargs = {"input_dict": keys_opt, "true_order": ["dates", dates]}
        if module == "ecalc_npv":
            values = np.asarray(function(pred_data, **kwargs)[0], dtype=float)  # type: ignore[operator,index]
        else:
            values = np.asarray(function(pred_data, kwargs), dtype=float)  # type: ignore[operator]
    return {
        "in_responses": np.concatenate([totals[v] for v in ("fopt", "fgpt", "fwpt", "fwit")]),
        "in_start": np.array(dates[0].date().isoformat()),  # type: ignore[attr-defined]
        "in_times": np.array([d.date().isoformat() for d in dates[1:]]),  # type: ignore[attr-defined]
        "in_prices": np.array([const[k] for k in const]),
        "in_price_names": np.array(list(const)),
        "in_stub": np.array([ECALC_STUB[k] for k in ECALC_STUB]),
        "in_stub_names": np.array(list(ECALC_STUB)),
        "in_source": np.array(source),
        "out_values": values,
    }


@case("popt_ecalc_npv", seed=20261421)
def popt_ecalc_npv(rng: np.random.Generator) -> dict[str, np.ndarray]:
    """PET's ``ecalc_npv`` (fd17979) with the eCalc stand-in :data:`ECALC_STUB`."""
    const = {"wop": 283.05, "wgp": 0.5, "wwp": 37.74, "wwi": 12.58, "wem": 60.0,
             "disc": 0.08, "obj_scaling": -1.0e6}  # fmt: skip
    return _ecalc_case(rng, "ecalc_npv", const)


@case("popt_ecalc_pareto_npv_wemc", seed=20261422)
def popt_ecalc_pareto_npv_wemc(rng: np.random.Generator) -> dict[str, np.ndarray]:
    """PET's ``ecalc_pareto_npv`` (fd17979), CO2 cost correction ``wemc``; eCalc stubbed."""
    const = {"wop": 283.05, "wgp": 0.5, "wwp": 37.74, "wwi": 12.58, "wem": 60.0,
             "disc": 0.08, "obj_scaling": -1.0e6, "wemc": 25.0}  # fmt: skip
    return _ecalc_case(rng, "ecalc_pareto_npv", const)


@case("popt_ecalc_pareto_npv_w", seed=20261423)
def popt_ecalc_pareto_npv_w(rng: np.random.Generator) -> dict[str, np.ndarray]:
    """PET's ``ecalc_pareto_npv`` (fd17979), emission-intensity weight ``w``; eCalc stubbed."""
    const = {"wop": 283.05, "wgp": 0.5, "wwp": 37.74, "wwi": 12.58, "wem": 60.0,
             "disc": 0.08, "obj_scaling": -1.0e6, "w": 0.3}  # fmt: skip
    return _ecalc_case(rng, "ecalc_pareto_npv", const)


@case("popt_ren_npv_co2", seed=20261424)
def popt_ren_npv_co2(rng: np.random.Generator) -> dict[str, np.ndarray]:
    """PET's ``ren_npv_co2`` (fd17979); eCalc stubbed (CO2 linear in the rates and wind).

    ``disc = 0`` and a stub that ignores dates and THP, so PET's two bugs (the per-step
    discount, the rates dated at the step's end) and APEX's THP at the step's end (PET's at
    its start) do not enter; the wind series is the same for every member (APEX's one series).
    """
    import tempfile

    import yaml

    n_times, ne = 6, 5
    dates = _npv_dates(n_times)
    totals = _ecalc_totals(rng, n_times, ne)
    thp = {f"wthp:{w}": rng.uniform(100.0, 300.0, (n_times, ne)) for w in ("inj1", "inj2")}
    wind_series = rng.uniform(0.0, 50.0, n_times)
    pred_data = [{v: np.zeros(ne) for v in totals} | {k: thp[k][0] for k in thp}] + [
        {v: totals[v][k] for v in totals} | {key: thp[key][k + 1] for key in thp}
        for k in range(n_times - 1)
    ]
    const = {"wop": 283.05, "wgp": 0.5, "wwp": 37.74, "wwi": 12.58, "wem": 60.0,
             "disc": 0.0, "obj_scaling": -1.0e6}  # fmt: skip
    keys_opt = {
        "npv_const": list(const.items()),
        "npv_kwargs": [("power", "power_arrays"), ("yamlfile", "ecalc_config")],
        "datatype": list(totals) + list(thp),
        "parallel": 1,
    }
    with tempfile.TemporaryDirectory() as folder, _stub_ecalc(Path(folder)):
        work = Path(folder)
        np.savez(work / "power_arrays.npz", wind=np.tile(wind_series, (ne, 1)))
        (work / "ecalc_config.yaml").write_text(
            yaml.safe_dump({"TIME_SERIES": [{"NAME": "SIM", "FILE": "ecalc_input.csv"}]})
        )
        namespace, source = _pet_npv_source("ren_npv_co2")
        values = namespace["ren_npv_co2"](pred_data, keys_opt, ["dates", dates])  # type: ignore[operator]
    return {
        "in_responses": np.concatenate([totals[v] for v in ("fopt", "fgpt", "fwpt", "fwit")]),
        "in_wthp": np.stack([thp[k][1:] for k in thp]),  # (n_injectors, n_times - 1, ne)
        "in_wind": wind_series[:-1],  # PET: power['wind'][:, :-1], one per step
        "in_start": np.array(dates[0].date().isoformat()),  # type: ignore[attr-defined]
        "in_times": np.array([d.date().isoformat() for d in dates[1:]]),  # type: ignore[attr-defined]
        "in_prices": np.array([const[k] for k in const]),
        "in_price_names": np.array(list(const)),
        "in_stub": np.array([ECALC_STUB[k] for k in ECALC_STUB]),
        "in_stub_names": np.array(list(ECALC_STUB)),
        "in_source": np.array(source),
        "out_values": np.asarray(values, dtype=float),
    }


#: PET-fork's Brugge MEC module. It is not part of the exported pet-source (it
#: lives in Examples/), and the published runs used the checkout's working copy, which has
#: uncommitted changes; the reference records the file's SHA-256.
MEC_MODULE = (
    Path(os.environ.get("PET_FORK_CHECKOUT", Path(__file__).resolve().parents[3] / "PET-fork"))
    / "Examples/.../Brugge/ml_model_error.py"
)


def _load_mec_module() -> tuple[object, str]:
    """Import PET-fork's ``ml_model_error.py``, stubbing ``torchviz`` (drawing only)."""
    import hashlib
    import types

    if not MEC_MODULE.is_file():
        sys.exit(f"{MEC_MODULE} not found; set PET_FORK_CHECKOUT to the PET-fork checkout")
    sys.modules.setdefault("torchviz", types.SimpleNamespace(make_dot=None))  # type: ignore[arg-type]
    spec = importlib.util.spec_from_file_location("ml_model_error", MEC_MODULE)
    if spec is None or spec.loader is None:
        sys.exit(f"cannot import {MEC_MODULE}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, hashlib.sha256(MEC_MODULE.read_bytes()).hexdigest()


@case("mec_lstm_forward", seed=20261026, source="fork")
def mec_lstm_forward(rng: np.random.Generator) -> dict[str, np.ndarray]:
    """PG-LSTM well residuals of PET-fork's ``LSTMWellModel``.

    Two wells as in the Brugge runs (a producer with three outputs, an injector with one),
    20 report times, hidden size 20. Features have Brugge-like magnitudes (block sizes,
    permeabilities in mD, densities, saturations, pressure differences), and parameters are
    drawn at the paper's prior scale N(0, 0.001^2) and at a larger one that drives the gates
    into their nonlinear range. Each member's parameters are assigned with PET-fork's
    ``assign_nn_params`` and its residuals computed with ``LSTMWellModel.forward``, as
    ``add_me_residual`` does.

    ``well_open`` is given as PET-fork's ``initialize_lstm_models`` builds it, a list of
    per-output flags per report time with some shut periods. ``forward`` tests the list's
    truth (``if not well_open[ts]``), which a non-empty list always passes, so the stored
    outputs are never zeroed: the parity test checks APEX's ``shut_wells = "ignore"``.
    """
    import torch

    module, digest = _load_mec_module()
    n_steps, hidden = 20, 20
    wells = {"producer": (3, 3), "injector": (2, 1)}  # (blocks, outputs)
    arrays: dict[str, np.ndarray] = {
        "in_n_steps": np.array(n_steps),
        "in_hidden_size": np.array(hidden),
        "in_module_sha256": np.array(digest),
    }
    for well, (n_blocks, n_out) in wells.items():
        # 12 features per block: DX DY DZ PERMX PERMY PERMZ PORO, OIL_DEN WAT_DEN SWAT SOIL
        # PRESS_DIFF (feature-major, then block), with zero padding for unperforated blocks.
        lows = [80, 80, 2, 10, 10, 1, 0.1, 50, 60, 0.1, 0.0, -300]
        highs = [120, 120, 20, 3000, 3000, 300, 0.3, 56, 63, 0.9, 0.9, 300]
        features = np.stack(
            [
                rng.uniform(lo, hi, size=(n_steps, n_blocks))
                for lo, hi in zip(lows, highs, strict=True)
            ],
            axis=1,
        ).reshape(n_steps, 12 * n_blocks)
        features[:4, -n_blocks] = 0.0  # one block perforated only from the fifth report on
        observed_max = rng.uniform(50, 3000, size=n_out)
        ranges = [[1e-4 * 0.0, 0.9999 * m] for m in observed_max]
        well_open = [[bool(t >= 3) or k == 1 for k in range(n_out)] for t in range(n_steps)]
        model = module.LSTMWellModel(  # type: ignore[attr-defined]
            well_open,
            obs_ranges=ranges,
            input_size=features.shape[1],
            hidden_size=hidden,
            output_size=n_out,
        )
        n_params = module.get_nn_params(model).size  # type: ignore[attr-defined]
        theta = np.concatenate(
            [rng.normal(0, 0.001, (n_params, 3)), rng.normal(0, 0.3, (n_params, 3))], axis=1
        )
        outputs = []
        for j in range(theta.shape[1]):
            module.assign_nn_params(model, theta[:, j])  # type: ignore[attr-defined]
            x = torch.tensor(features.tolist()).unsqueeze(0).float()
            sim_obs = np.zeros((n_steps, n_out))
            with torch.no_grad():
                out = model(x, sim_obs, well_open)
            outputs.append(out.detach().numpy().reshape(n_steps, n_out))
        arrays |= {
            f"in_{well}_features": features,  # (L, I), shared by the members
            f"in_{well}_theta": theta,  # (n_params, n): APEX layout, ensemble last
            f"in_{well}_scale": np.array([r[1] for r in ranges]),
            f"in_{well}_open": np.array(well_open),
            f"out_{well}_residuals": np.stack(outputs, axis=-1).astype(np.float64),  # (L, O, n)
        }
    return arrays


@case("mec_brugge_features", seed=20261027, source="fork")
def mec_brugge_features(rng: np.random.Generator) -> dict[str, np.ndarray]:
    """PET-fork's PG-LSTM inputs and residuals on its Brugge reference run.

    ``ModelErrorProd`` (its defaults: the 20 producers and 10 injectors, WOPR/WBHP/WWCT and
    WBHP, hidden size 20) reads the schedule ``include/Hist`` and the Eclipse run
    ``model_error/BRUGGE_REF_FAULTS`` of the PET-fork checkout; each well's network is
    given parameters drawn N(0, 0.05^2) and returns its input sequence and residuals
    (``predict_residuals_lstm``). The opt-in test ``tests/parity/test_mec_brugge.py``
    reads the same run with APEX and compares.
    """
    import torch  # noqa: F401  (PET-fork's model)

    module, digest = _load_mec_module()
    brugge = MEC_MODULE.parent
    previous = Path.cwd()
    os.chdir(brugge)  # ModelErrorProd reads its files relative to the case folder
    try:
        model = module.ModelErrorProd(  # type: ignore[attr-defined]
            case_file="./model_error/BRUGGE_REF_FAULTS",
            schedule_file="./include/Hist",
            schedule_kws=["COMPDAT", "WCONHIST", "WCONINJH"],
            obs_data="./obs_var.npz",
            hidden_size=20,
        )
        arrays: dict[str, np.ndarray] = {"in_module_sha256": np.array(digest)}
        for well in model.prod_wells + model.inj_wells:
            n_out = len(model.prod_data if well in model.prod_wells else model.inj_data)
            theta = rng.normal(0, 0.05, len(model.me_model[well]["params"])).astype(np.float32)
            residuals, inputs = model.predict_residuals_lstm(
                well, theta.astype(np.float64), np.zeros((20, n_out))
            )
            arrays |= {
                f"in_{well}_theta": theta,
                f"in_{well}_scale": np.array(model.me_model[well]["obs_ranges"])[:, 1],
                f"out_{well}_blocks": np.array(model.me_model[well]["all_blocks"]),
                # Per report time and output; WBHP counts as open (initialize_lstm_models).
                f"out_{well}_open": np.array(model.me_model[well]["well_open"], dtype=bool),
                f"out_{well}_features": inputs.numpy()[0].astype(np.float64),
                f"out_{well}_residuals": residuals.detach().numpy()[0].astype(np.float64),
            }
    finally:
        os.chdir(previous)
    return arrays


def generate(name: str) -> Path:
    """Run case ``name`` and write its ``.npz`` file."""
    entry = CASES[name]
    if SOURCES[entry.source] != PET_COMMIT:
        raise SystemExit(
            f"case {name!r} needs PET source {entry.source!r} at {SOURCES[entry.source]}, "
            f"but this interpreter provides {PET_COMMIT} from {PET_ROOT}"
        )
    arrays = entry.function(np.random.default_rng(entry.seed))
    bad_keys = [key for key in arrays if not key.startswith(("in_", "out_"))]
    if bad_keys:
        raise ValueError(f"case {name!r} returned keys without in_/out_ prefix: {bad_keys}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"{name}.npz"
    np.savez(
        path,
        **arrays,
        meta_pet_source=np.array(entry.source),
        meta_pet_commit=np.array(PET_COMMIT),
        meta_numpy_version=np.array(np.__version__),
        meta_seed=np.array(entry.seed),
    )
    return path


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "cases", nargs="*", help="case names to generate (default: all of this source)"
    )
    parser.add_argument("--list", action="store_true", help="list available cases and exit")
    args = parser.parse_args(argv)

    if args.list:
        for name in sorted(CASES):
            source = CASES[name].source
            runnable = "" if SOURCES[source] == PET_COMMIT else "  (other source)"
            print(f"{name}  [{source}]{runnable}")
        return 0
    unknown = sorted(set(args.cases) - set(CASES))
    if unknown:
        parser.error(f"unknown cases: {', '.join(unknown)}; see --list")
    names = args.cases or sorted(n for n, c in CASES.items() if SOURCES[c.source] == PET_COMMIT)
    if not names:
        parser.error(f"no cases for the PET commit {PET_COMMIT} this interpreter provides")
    for name in names:
        print(f"wrote {generate(name)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
