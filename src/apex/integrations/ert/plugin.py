"""APEX's ERT plugin: two forward-model steps and four workflow jobs.

The steps are ``APEX_RENDER_DECK`` and ``APEX_WRITE_PARAMETERS``; the jobs ``APEX_CV``,
``APEX_CV_SELECT``,
``APEX_HISTORY_MATCH`` and ``APEX_UPDATE``.

ERT finds it through the entry point ``[project.entry-points.ert] apex``. Besides this
module, only :func:`apex.integrations.ert.case.read_ert_case` imports ERT (the ``ert``
extra).
"""

import json
import logging
import math
import sys
from pathlib import Path
from typing import Any

import ert
import numpy as np
from ert import ErtScript, ForwardModelStepDocumentation, ForwardModelStepPlugin

from apex.domain.observations import DataKey
from apex.integrations.ert import cv
from apex.integrations.ert.update import (
    ErtUpdateInput,
    apex_update,
    read_update_config,
)

logger = logging.getLogger(__name__)

#: ERT's observation selection must be off for APEX_UPDATE: no outlier rejection beyond
#: this many standard deviations, and no spread cutoff above this.
MIN_ENKF_ALPHA = 1e3
MAX_STD_CUTOFF = 1e-12


class ApexRenderDeck(ForwardModelStepPlugin):
    """Render APEX's Mako deck in the run path from the fields ERT wrote there."""

    def __init__(self) -> None:
        """Run ``apex.integrations.ert.render`` with this Python (APEX's environment)."""
        super().__init__(
            name="APEX_RENDER_DECK",
            command=[
                sys.executable,
                "-m",
                "apex.integrations.ert.render",
                "--export",
                "<EXPORT>",
                "--output",
                "<OUTPUT>",
                "--realization",
                "<IENS>",
            ],
            required_keywords=["<EXPORT>", "<OUTPUT>"],
        )

    @staticmethod
    def documentation() -> ForwardModelStepDocumentation | None:
        """Describe the step for ERT's documentation."""
        return ForwardModelStepDocumentation(
            category="simulators.reservoir",
            description=(
                "Renders an APEX case's Mako deck (apex_export.json, written by "
                "``apex ert export``) into <OUTPUT>, from the FIELD files ERT wrote in the "
                "run path, so ERT runs the same deck as APEX."
            ),
            examples=(
                "FORWARD_MODEL APEX_RENDER_DECK(<EXPORT>=<CONFIG_PATH>/apex_export.json, "
                "<OUTPUT>=<ECLBASE>.DATA)"
            ),
        )


class ApexTransform(ForwardModelStepPlugin):
    """Apply APEX's data transforms to the simulated summary."""

    def __init__(self) -> None:
        """Run ``apex.integrations.ert.transform_step`` with this Python."""
        super().__init__(
            name="APEX_TRANSFORM",
            command=[
                sys.executable, "-m", "apex.integrations.ert.transform_step",
                "--export", "<EXPORT>", "--output", "<OUTPUT>",
            ],
            required_keywords=["<EXPORT>", "<OUTPUT>"],
        )  # fmt: skip

    @staticmethod
    def documentation() -> ForwardModelStepDocumentation | None:
        """Describe the step for ERT's documentation."""
        return ForwardModelStepDocumentation(
            category="utility.apex",
            description=(
                "Reads the summary at an exported case's raw keys and writes APEX's "
                "transformed data (e.g. wavelet coefficients) as GEN_DATA, so ERT assimilates "
                "what APEX does."
            ),
            examples=(
                "FORWARD_MODEL APEX_TRANSFORM(<EXPORT>=<CONFIG_PATH>/apex_export.json, "
                "<OUTPUT>=apex_transformed_0.txt)"
            ),
        )


class ApexWriteParameters(ForwardModelStepPlugin):
    """Write APEX's parameters in the run path as ERT would (B2's parameter step)."""

    def __init__(self) -> None:
        """Run ``apex.integrations.ert.parameters`` with this Python (APEX's environment)."""
        super().__init__(
            name="APEX_WRITE_PARAMETERS",
            command=[
                sys.executable, "-m", "apex.integrations.ert.parameters",
                "--case", "<CASE>", "--realization", "<IENS>",
            ],
            required_keywords=["<CASE>"],
        )  # fmt: skip

    @staticmethod
    def documentation() -> ForwardModelStepDocumentation | None:
        """Describe the step for ERT's documentation."""
        return ForwardModelStepDocumentation(
            category="utility.apex",
            description=(
                "Writes the GEN_KW, SURFACE and FIELD values APEX left in the run path "
                "(apex_parameters.npz) as ERT itself writes them: parameters.json and .txt, "
                "templates, surfaces, fields with their transforms. APEX puts it "
                "first in APEX_HISTORY_MATCH's batches."
            ),
            examples="FORWARD_MODEL APEX_WRITE_PARAMETERS(<CASE>=<CONFIG_PATH>/<CONFIG_FILE>)",
        )


@ert.plugin(name="apex")
def installable_forward_model_steps() -> list[type[ForwardModelStepPlugin]]:
    """Return APEX's forward-model steps."""
    return [ApexRenderDeck, ApexWriteParameters, ApexTransform]


class ApexCV(ErtScript):
    """Score an iterate's held-out data with APEX's CV criterion (hook: POST_SIMULATION).

    Argument: the case's ``apex_export.json``.
    """

    def run(self, run_paths: Any, ensemble: Any, workflow_args: list[str]) -> str:  # noqa: ANN401
        """Score the realizations of ``ensemble``'s iteration; record them in apex_cv.json."""
        case_dir = Path(workflow_args[0]).resolve().parent
        iteration = int(ensemble.iteration)
        paths = [Path(p) for p in run_paths.get_paths(range(ensemble.ensemble_size), iteration)]
        entry = cv.score_iteration(case_dir, iteration, [p for p in paths if p.exists()])
        return f"APEX_CV iteration {iteration}: " + ", ".join(
            f"{role} {values['mean']}" for role, values in entry.items()
        )


class ApexCVSelect(ErtScript):
    """Pick the posterior by APEX's stopping rule (hook: POST_EXPERIMENT).

    Arguments: the case's ``apex_export.json``, the rule (``select`` or ``patience``),
    and for ``patience`` its patience.
    """

    def run(self, workflow_args: list[str]) -> str:
        """Apply the rule to the recorded CV scores; record the selection."""
        case_dir = Path(workflow_args[0]).resolve().parent
        rule = workflow_args[1] if len(workflow_args) > 1 else "select"
        patience = int(workflow_args[2]) if len(workflow_args) > 2 else 2
        selection = cv.select(case_dir, rule, patience)
        return f"APEX_CV_SELECT: {selection}"


class ApexHistoryMatch(ErtScript):
    """History-match the case with APEX's methods.

    Arguments: the case's ``.ert`` file and, optionally, the ``apex.toml`` (default: next to
    the case). APEX runs its loop and ERT the forward model of every forecast; the run goes
    to ``<case>/apex_runs/<name>/``, and a run already there is resumed. Its iterates are
    then added to ERT's storage as the experiment ``apex:<name>``, for ERT's GUI. With
    ``[methods.<label>]`` each method is a run of its own (``apex:<run>:<label>``); a failed
    method fails the job after the others ran.
    """

    #: ``ert workflow`` reports a failed run as a failure, not as a logged error.
    stop_on_fail = True

    def run(self, storage: Any, workflow_args: list[str]) -> str:  # noqa: ANN401
        """Run or continue APEX's history matching of the case; write it to ``storage``."""
        from apex.integrations.ert.case import history_match_all

        if not workflow_args:
            raise ValueError("APEX_HISTORY_MATCH needs the case: <CONFIG_PATH>/<CONFIG_FILE>")
        apex_file = Path(workflow_args[1]) if len(workflow_args) > 1 else None
        # ERT holds its storage open for the workflow: the runs are written back through it.
        results = history_match_all(Path(workflow_args[0]), apex_file, storage=storage)
        return "APEX_HISTORY_MATCH: " + "; ".join(
            f"{label + ' ' if label else ''}{'complete' if r.complete else 'stopped'} after "
            f"{len(r.history)} forecasts"
            for label, r in results.items()
        )


class ApexDiagnose(ErtScript):
    """APEX's diagnosis of a run, for ERT users (rules, no simulator runs).

    Argument: an APEX run directory (``apex_runs/<name>``), or an exported case's folder
    (``<CONFIG_PATH>``), whose ERT run is imported first. Writes ``agents/diagnosis.md`` in
    the run directory.
    """

    def run(self, workflow_args: list[str]) -> str:
        """Diagnose the run; return where the report went."""
        from apex.integrations.ert.agents import diagnose

        if not workflow_args:
            raise ValueError("APEX_DIAGNOSE needs a run directory or an exported case's folder")
        return f"APEX_DIAGNOSE: {diagnose(Path(workflow_args[0]))}"


class ApexReport(ErtScript):
    """APEX's narrative report of a run, for ERT users (rules).

    Argument: as ``APEX_DIAGNOSE``. Writes ``report/narrative.md`` in the run directory.
    """

    def run(self, workflow_args: list[str]) -> str:
        """Write the report; return where it went."""
        from apex.integrations.ert.agents import report

        if not workflow_args:
            raise ValueError("APEX_REPORT needs a run directory or an exported case's folder")
        return f"APEX_REPORT: {report(Path(workflow_args[0]))}"


class ApexUpdate(ErtScript):
    """Replace ERT's posterior with APEX's analysis (hook: POST_UPDATE).

    Argument: the ``apex.toml`` with ``[method]`` (``es`` or ``esmda``, its strategy and
    localization) and optionally ``[run] seed``. ERT's ES or ES-MDA runs as usual; after
    each update, this job recomputes the posterior from the prior with APEX and overwrites
    what ERT wrote. Records go to ``apex_updates/<experiment id>/`` next to the
    ``apex.toml``.
    """

    #: A failed APEX update stops ERT: going on would leave ERT's posterior in place.
    stop_on_fail = True

    def run(
        self,
        storage: Any,  # noqa: ANN401 (ERT's Storage)
        ensemble: Any,  # noqa: ANN401 (the prior)
        observation_settings: Any,  # noqa: ANN401
        random_seed: int,
        workflow_args: list[str],
    ) -> str:
        """Update the posterior of ``ensemble`` (the prior) with APEX's analysis."""
        if not workflow_args:
            raise ValueError("APEX_UPDATE needs the apex.toml: <CONFIG_PATH>/apex.toml")
        apex_file = Path(workflow_args[0]).resolve()
        config = read_update_config(apex_file)
        _check_observation_settings(observation_settings)
        prior = ensemble
        posterior = _posterior_of(storage, prior)
        update, realizations = _update_input(prior)
        seed = config.seed if config.seed is not None else int(random_seed)
        record = apex_file.parent / "apex_updates" / str(prior.experiment.id)
        record.mkdir(parents=True, exist_ok=True)
        prepared_file = record / "prepared.npz"
        prepared = None
        if update.iteration > 0 and prepared_file.exists():
            with np.load(prepared_file) as stored:
                prepared = {k: stored[k] for k in stored.files}
        result, prepared_now = apex_update(update, config, seed, prepared)
        if update.iteration == 0 and prepared_now:
            np.savez(prepared_file, **prepared_now)  # type: ignore[arg-type]
        for group, values in result.items():
            posterior.save_parameters_numpy(values, group, realizations)
        summary = {
            "prior": prior.name, "posterior": posterior.name, "iteration": update.iteration,
            "inflation": update.inflation, "n_data": len(update.keys),
            "realizations": realizations.tolist(), "seed": seed,
            "method": config.method.model_dump(mode="json"),
        }  # fmt: skip
        (record / f"update_{update.iteration}.json").write_text(json.dumps(summary, indent=2))
        return (
            f"APEX_UPDATE: {posterior.name} from {prior.name} by APEX ({config.method.name}, "
            f"alpha {update.inflation:g}, {len(update.keys)} data)"
        )


def _check_observation_settings(settings: Any) -> None:  # noqa: ANN401
    outlier = settings.outlier_settings
    if (
        float(outlier.alpha) < MIN_ENKF_ALPHA
        or float(outlier.std_cutoff) > MAX_STD_CUTOFF
        or settings.auto_scale_observations
    ):
        # APEX's analysis assimilates every datum with its own error; ERT's selection
        # (outliers, small spreads, auto-scaling) would make the two use different data.
        raise ValueError(
            "APEX_UPDATE needs ERT's observation selection off: ENKF_ALPHA >= 1e3 (e.g. 1e6), "
            "STD_CUTOFF <= 1e-12 (e.g. 1e-30), and no AUTO_SCALE_OBSERVATIONS"
        )


def _posterior_of(storage: Any, prior: Any) -> Any:  # noqa: ANN401
    """Return the ensemble ERT just updated from ``prior`` (the newest with it as parent)."""
    children = [e for e in prior.experiment.ensembles if e.parent == prior.id]
    if not children:
        raise ValueError(f"no posterior of {prior.name}: hook APEX_UPDATE at POST_UPDATE")
    return max(children, key=lambda e: e.started_at)


def _update_input(prior: Any) -> tuple[ErtUpdateInput, np.ndarray]:  # noqa: ANN401
    """Read the prior's parameters, responses and observations from ERT's storage."""
    from ert.config import Field
    from ert.config.analysis_config import parse_es_mda_weights
    from ert.storage.local_experiment import ExperimentType

    experiment = prior.experiment
    kind = experiment.experiment_type
    if kind == ExperimentType.ES_MDA:
        schedule = tuple(float(w) for w in parse_es_mda_weights(experiment.relative_weights))
    elif kind == ExperimentType.ENSEMBLE_SMOOTHER:
        schedule = (1.0,)
    else:
        raise ValueError(f"APEX_UPDATE runs in ES and ES-MDA experiments, not {kind}")
    realizations = np.flatnonzero(prior.get_realization_mask_with_responses())
    frame = prior.get_observations_and_responses(experiment.observation_keys, realizations).sort(
        by=["observation_key", "index"]
    )
    columns = [str(i) for i in realizations]
    predicted = frame.select(columns).to_numpy().astype(np.float64)
    complete = np.isfinite(predicted).all(axis=1)
    if not complete.all():
        # As ERT, data a realization has no response for are left out of the update.
        logger.warning("APEX_UPDATE: %d data without every response are left out",
                       int((~complete).sum()))  # fmt: skip
    keys = [
        DataKey(str(index)[:10], str(key))
        for index, key in zip(frame["index"], frame["response_key"], strict=True)
    ]
    parameters = {}
    bounds = {}
    for group in experiment.update_parameters:
        parameters[group] = prior.load_parameters_numpy(group, realizations).astype(np.float64)
        config = experiment.parameter_configuration[group]
        lower, upper = -math.inf, math.inf
        if isinstance(config, Field):
            lower = -math.inf if config.truncation_min is None else float(config.truncation_min)
            upper = math.inf if config.truncation_max is None else float(config.truncation_max)
        bounds[group] = (lower, upper)
    update = ErtUpdateInput(
        parameters=parameters,
        bounds=bounds,
        predicted=np.ascontiguousarray(predicted[complete]),
        observations=frame["observations"].to_numpy().astype(np.float64)[complete],
        errors=frame["std"].to_numpy().astype(np.float64)[complete],
        keys=[k for k, ok in zip(keys, complete, strict=True) if ok],
        schedule=schedule,
        iteration=int(prior.iteration),
    )
    return update, realizations


@ert.plugin(name="apex")
def ertscript_workflow(config: Any) -> None:  # noqa: ANN401 (ERT's WorkflowConfigs)
    """Register APEX's workflow jobs."""
    config.add_workflow(ApexCV, "APEX_CV", category="observations.apex")
    config.add_workflow(ApexCVSelect, "APEX_CV_SELECT", category="observations.apex")
    config.add_workflow(ApexHistoryMatch, "APEX_HISTORY_MATCH", category="analysis.apex")
    config.add_workflow(ApexUpdate, "APEX_UPDATE", category="analysis.apex")
    config.add_workflow(ApexDiagnose, "APEX_DIAGNOSE", category="analysis.apex")
    config.add_workflow(ApexReport, "APEX_REPORT", category="analysis.apex")
