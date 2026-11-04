"""History matching with a surrogate: the surrogate's model, checks and refits.

:class:`SurrogateForecasting` is what a run with ``[surrogate]`` holds beyond the plain
loop (``workflows/history_matching.py`` asks it at each forecast):

- the **active model** (the original, or the latest refit) as a
  :class:`~apex.simulation.surrogate.SurrogateSimulator` and the local executor its
  forecasts run on;
- when to **check** (:meth:`checks_at`: iterations ``check_every``, ``2 check_every``,
  ...), with which tolerances (the model's gate, tightened by ``sigma_ratio``), and the
  checks done;
- the **refits** (:meth:`refit`): the checked runs join the snapshot set, the plugin is
  fitted again from the stream ``surrogate`` and gated on the held-out runs plus a share of
  the new ones, and written to ``surrogates/refit_<k>/`` in the run directory with its
  snapshot set, so that a refit is reproducible from the run directory alone;
- the **bounds**: the surrogate evaluations counted against ``max_evaluations``, the
  simulator runs taken, and whether the run switched to the simulator;
- the **cost** (:func:`forecast_timing`, decision 14): the surrogate's and the simulator's
  forecast times, from the run's ``forecast_done`` events, against the build's full runs.

:meth:`state` and :meth:`restore` carry all of it through ``state.json``, so a resume is
bit-identical to an uninterrupted run, across checks and refits: predictions are
deterministic, and refits draw only from the checkpointed stream.
"""

import logging
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from apex.core.errors import ConfigError, WorkflowError
from apex.core.simulation import Executor, Simulator
from apex.core.surrogates import EarlyStartSurrogate, SurrogateModel
from apex.domain.config import SurrogateRunSection
from apex.domain.observations import DataKey
from apex.domain.surrogate import (
    BuildCost,
    CoarseSection,
    DomainExtent,
    FitRecord,
    GateResult,
    GateTolerances,
    QuantityKind,
    SnapshotReference,
    SnapshotSet,
    SurrogateManifest,
    TrainingDomain,
)
from apex.simulation.surrogate import SurrogateSimulator
from apex.storage.surrogate_directory import SNAPSHOTS, SurrogateDirectory, load_snapshots
from apex.storage.surrogate_files import write_surrogate
from apex.workflows.surrogate import fit_and_gate

logger = logging.getLogger(__name__)

#: A refit's folder in the run directory, by the iteration that made it.
REFITS = Path("surrogates")


class SurrogateCheckFailed(WorkflowError):
    """A check of the surrogate against the simulator failed, and the run stops (resumable)."""


@dataclass(frozen=True)
class ActiveModel:
    """A fitted surrogate the run forecasts with.

    Attributes
    ----------
    path
        Its manifest (an original's as configured; a refit's inside the run directory).
    identity
        The SHA-256 of its files.
    manifest
        Its manifest.
    simulator
        The model as a simulator (given properties, not the state).
    """

    path: Path
    identity: str
    manifest: SurrogateManifest
    simulator: SurrogateSimulator

    @property
    def model(self) -> SurrogateModel:
        """The fitted plugin."""
        return self.simulator.fitted()[0]

    @property
    def domain(self) -> TrainingDomain:
        """Its validated domain."""
        return self.simulator.fitted()[1]


#: Loads a model's files as an :class:`ActiveModel`, refusing another identity (``None``: any).
ModelLoader = Callable[[Path, str | None], ActiveModel]


class SurrogateForecasting:
    """A run's surrogate: its active model, checks, refits and bounds.

    Parameters
    ----------
    section
        ``[surrogate]``.
    original
        The configured model.
    executor
        Where the surrogate's forecasts run (local: serial, or process with ``workers``).
    tolerances
        The checks' (and refits') gate: the model's own, tightened by ``sigma_ratio``.
    variances
        ``(n_model_keys,)``, the observation errors' variances at the model's keys, for a
        refit's gate (``None``: relative errors, as the model was gated).
    kinds
        The kinds of quantity the model's gate used, for a refit's.
    loader
        Loads a model's files (a refit's, or the original's on resume).
    plugin
        Creates a fresh, unfitted instance of the model's plugin with its options.
    stored_options
        The plugin's options as a refit's manifest in a given folder records them (options
        naming files relative to that folder); ``None``: the original's manifest's as they are
        (right only for a refit as deep below the run as the original's model; tests).
    snapshots
        Returns the snapshot set the original was fitted and gated on (digest-checked);
        ``None`` when the model records none (refits refused at validation then).
    cost
        What the original's build's full runs cost (``None``: unknown), for the run's
        timing report.
    coarse
        A coarse-corrected model's coarse stack: takes a member's state, as the
        run's simulator does, and runs on the run's executor before each surrogate forecast.
    stored_coarse
        The coarse stack as a refit's manifest in a given folder records it.
    """

    def __init__(
        self,
        section: SurrogateRunSection,
        *,
        original: ActiveModel,
        executor: Executor,
        tolerances: GateTolerances,
        variances: NDArray[np.float64] | None,
        kinds: Mapping[str, QuantityKind],
        loader: ModelLoader,
        plugin: Callable[[], SurrogateModel],
        snapshots: Callable[[], SnapshotSet] | None,
        cost: BuildCost | None = None,
        stored_options: Callable[[Path], dict[str, object]] | None = None,
        coarse: Simulator | None = None,
        stored_coarse: Callable[[Path], CoarseSection] | None = None,
    ) -> None:
        self.section = section
        self.original = original
        self.active = original
        self.executor = executor
        self.tolerances = tolerances
        self.variances = variances
        self.kinds = dict(kinds)
        self.loader = loader
        self.plugin = plugin
        self.stored_options = stored_options
        self.snapshots = snapshots
        self.cost = cost
        self.coarse = coarse
        self.stored_coarse = stored_coarse
        # Loop state, checkpointed (state.json "surrogate").
        self.refits: list[dict[str, Any]] = []
        self.checks: list[dict[str, Any]] = []
        # Accepted iterates forecast again after a change of source: {iteration, accepted,
        # source}; cross validation compares only iterates from the latest one on.
        self.rescores: list[dict[str, Any]] = []
        self.evaluations = 0
        self.simulator_runs = 0
        # A coarse-corrected model's coarse runs: simulator runs of their own.
        self.coarse_runs = 0
        # An early-start model's early runs: each member simulated up to
        # the model's early date before it is predicted; simulator runs of their own.
        self.early_runs = 0
        self.switched_at: int | None = None

    # ------------------------------------------------------------------ decisions

    @property
    def early_until(self) -> str | None:
        """The report date members are simulated up to for an early-start model, else ``None``."""
        model = self.original.model
        return model.early_until if isinstance(model, EarlyStartSurrogate) else None

    @property
    def extent(self) -> DomainExtent:
        """The original domain's shape beyond its box, kept by refits."""
        return self.original.domain.extent

    def checks_at(self, iteration: int) -> bool:
        """Whether the forecast of ``iteration`` is checked (iterations k >= 1, k % every == 0)."""
        every = self.section.check_every
        return every > 0 and iteration >= 1 and iteration % every == 0

    def check_count(self, available: int) -> int:
        """Members checked among ``available`` forecast by the surrogate."""
        wanted = self.section.check_members
        return available if wanted == "all" else min(int(wanted), available)

    def charge(self, n: int, iteration: int) -> None:
        """Count ``n`` surrogate evaluations against ``max_evaluations``.

        Raises
        ------
        WorkflowError
            If they would exceed the bound: the run stops at its last checkpoint.
        """
        limit = self.section.max_evaluations
        if self.evaluations + n > limit:
            logger.error(
                "surrogate evaluations exhausted",
                extra={"event": "surrogate_budget_exhausted", "iteration": iteration,
                       "evaluations": self.evaluations, "asked": n, "max_evaluations": limit},
            )  # fmt: skip
            raise WorkflowError(
                f"iteration {iteration}: {n} more surrogate evaluations would exceed "
                f"[surrogate] max_evaluations = {limit} ({self.evaluations} made); the run "
                "stops at its last checkpoint"
            )
        self.evaluations += n

    # ------------------------------------------------------------------ refits

    def _base(self) -> tuple[SnapshotSet, list[int]]:
        """Return the snapshot set of the active model and its held-out run ids."""
        if not self.refits:
            if self.snapshots is None:
                raise WorkflowError("the surrogate records no snapshot set: it cannot be refitted")
            base = self.snapshots()
        else:
            path = self.active.path.parent.parent / SNAPSHOTS
            base = load_snapshots(path)
            if base.digest() != self.refits[-1]["snapshots"]:
                raise WorkflowError(f"{path} is not the snapshot set the run recorded")
        gate = self.active.manifest.gate
        if gate is None:
            raise WorkflowError("the surrogate records no gate: it cannot be refitted")
        return base, list(gate.held_out)

    def refit(
        self,
        iteration: int,
        inputs: Mapping[str, NDArray[np.float64]],
        outputs: NDArray[np.float64],
        member_ids: Sequence[int],
        rng: np.random.Generator,
        run_path: Path,
        coarse_outputs: NDArray[np.float64] | None = None,
    ) -> GateResult:
        """Refit with the checked runs added; make the refit the active model if it passes.

        Parameters
        ----------
        inputs
            ``{variable: (size, n)}``: the checked members' properties (the model's inputs).
        outputs
            ``(n_model_keys, n)``: their simulator runs at the model's keys.
        member_ids
            ``(n,)``, the members (recorded; the runs get new ids in the snapshot set).
        rng
            The run's stream ``surrogate``: the split of the new runs, then the fit.
        coarse_outputs
            ``(n_model_keys, n)``: their coarse stack's runs, for a coarse-corrected model.

        Returns the refit's gate. The refit is written whether or not it passes.

        Raises
        ------
        WorkflowError
            If the refit's gate fails (the run stops), or the snapshots cannot be read.
        """
        base, held = self._base()
        n = outputs.shape[1]
        if (base.coarse_outputs is None) != (coarse_outputs is None):
            raise WorkflowError(
                "a coarse-corrected surrogate is refitted with the checked members' coarse "
                "runs, and only such a surrogate"
            )
        order = rng.permutation(n)
        n_held = 0 if n < 2 else min(n - 1, max(1, round(self.section.refit_held_out * n)))
        first = max(base.run_ids) + 1
        new_ids = tuple(first + j for j in range(n))
        combined = SnapshotSet(
            layout=base.layout,
            inputs={
                name: np.concatenate([base.inputs[name], np.asarray(inputs[name])], axis=1)
                for name in base.layout.names
            },
            keys=base.keys,
            outputs=np.concatenate([base.outputs, outputs], axis=1),
            run_ids=(*base.run_ids, *new_ids),
            provenance=base.provenance,
            failed=base.failed,
            coarse_outputs=None
            if base.coarse_outputs is None or coarse_outputs is None
            else np.concatenate([base.coarse_outputs, coarse_outputs], axis=1),
            # The build's coarse-only members ([design] coarse_only) stay with the refit.
            coarse_only_inputs=base.coarse_only_inputs,
            coarse_only_outputs=base.coarse_only_outputs,
        )
        held_ids = set(held) | {new_ids[int(j)] for j in order[:n_held]}
        held_positions = [p for p, r in enumerate(combined.run_ids) if r in held_ids]
        train_positions = [p for p, r in enumerate(combined.run_ids) if r not in held_ids]
        manifest = self.active.manifest
        model = self.plugin()
        try:
            fitted = fit_and_gate(
                model, combined.columns(train_positions), combined.columns(held_positions), rng,
                tolerances=self.tolerances, variances=self.variances, kinds=self.kinds,
                extent=self.extent, failed=list(combined.failed),
            )  # fmt: skip
        except (TypeError, ValueError, ArithmeticError, LookupError) as error:
            raise WorkflowError(
                f"iteration {iteration}: the surrogate {manifest.plugin!r} cannot be refitted: "
                f"{type(error).__name__}: {error}"
            ) from error
        folder = run_path / REFITS / f"refit_{iteration}"
        SurrogateDirectory(folder).save_snapshots(combined)
        provenance = combined.provenance.model_copy(
            update={"plan": "refit", "source": f"history matching, iteration {iteration}"}
        )
        model_file = folder / "model" / "surrogate.json"
        stored = write_surrogate(
            model_file,
            plugin=manifest.plugin,
            model=fitted.model,
            domain=fitted.domain,
            provenance=provenance,
            options=(
                manifest.options
                if self.stored_options is None
                else self.stored_options(model_file.parent)
            ),
            gate=fitted.gate,
            snapshots=SnapshotReference(path=f"../{SNAPSHOTS}", digest=combined.digest()),
            fit=FitRecord(stream_state=fitted.stream_state),
            apex_commit=manifest.apex_commit,
            coarse=(
                manifest.coarse
                if self.stored_coarse is None
                else self.stored_coarse(model_file.parent)
            ),
        )
        SurrogateDirectory(folder).write_json(
            "runs.json",
            {"iteration": iteration,
             "runs": {str(r): int(m) for r, m in zip(new_ids, member_ids, strict=True)}},
        )  # fmt: skip
        logger.info(
            "surrogate refitted",
            extra={"event": "surrogate_refit", "iteration": iteration,
                   "identity": stored.identity, "passed": fitted.gate.passed,
                   "runs": n, "held_out": n_held, "failures": fitted.gate.failures()},
        )  # fmt: skip
        if not fitted.gate.passed:
            raise WorkflowError(
                f"iteration {iteration}: the refitted surrogate fails its gate ("
                + "; ".join(fitted.gate.failures())
                + f"); it is in {folder}, the run stops at its last checkpoint"
            )
        self.refits.append({
            "iteration": iteration, "path": str(folder.relative_to(run_path) / "model" /
                                                 "surrogate.json"),
            "identity": stored.identity, "snapshots": combined.digest(),
        })  # fmt: skip
        self.active = self.loader(stored.path, stored.identity)
        return fitted.gate

    # ------------------------------------------------------------------ checkpoints

    def state(self) -> dict[str, Any]:
        """Return the loop state for ``state.json`` (JSON values)."""
        return {
            "original": {"path": str(self.original.path), "identity": self.original.identity},
            "active": self.active.identity,
            "refits": [dict(r) for r in self.refits],
            "checks": [dict(c) for c in self.checks],
            "rescores": [dict(r) for r in self.rescores],
            "evaluations": self.evaluations,
            "simulator_runs": self.simulator_runs,
            "switched_at": self.switched_at,
            **({"coarse_runs": self.coarse_runs} if self.coarse is not None else {}),
            **({"early_runs": self.early_runs} if self.early_until is not None else {}),
        }

    def restore(self, saved: Mapping[str, Any] | None, run_path: Path) -> None:
        """Continue from :meth:`state`, loading the active model with its recorded identity.

        Raises
        ------
        WorkflowError
            If the run was started without ``[surrogate]``.
        ConfigError
            If a model file's identity differs from the one the run recorded.
        """
        if saved is None:
            raise WorkflowError(
                "the run was started without [surrogate]; it cannot be resumed with it"
            )
        recorded = str(saved["original"]["identity"])
        if recorded != self.original.identity:
            raise ConfigError(
                f"the surrogate {self.original.path} has identity "
                f"{self.original.identity[:12]}..., not the {recorded[:12]}... the run was "
                "started with: the model file changed; resume refuses it"
            )
        self.refits = [dict(r) for r in saved.get("refits", [])]
        self.checks = [dict(c) for c in saved.get("checks", [])]
        self.rescores = [dict(r) for r in saved.get("rescores", [])]
        self.evaluations = int(saved.get("evaluations", 0))
        self.simulator_runs = int(saved.get("simulator_runs", 0))
        self.coarse_runs = int(saved.get("coarse_runs", 0))
        self.early_runs = int(saved.get("early_runs", 0))
        switched = saved.get("switched_at")
        self.switched_at = None if switched is None else int(switched)
        if self.refits:
            latest = self.refits[-1]
            self.active = self.loader(run_path / str(latest["path"]), str(latest["identity"]))
        else:
            self.active = self.original

    def describe(self) -> dict[str, Any]:
        """Return the run's surrogate facts for ``summary.json`` (without the posterior)."""
        return {
            "model": str(self.original.path),
            "plugin": self.original.manifest.plugin,
            "identity": self.original.identity,
            "active_identity": self.active.identity,
            "refits": [dict(r) for r in self.refits],
            "checks": [dict(c) for c in self.checks],
            "rescores": [dict(r) for r in self.rescores],
            "evaluations": self.evaluations,
            "max_evaluations": self.section.max_evaluations,
            "switched_to_simulator_at": self.switched_at,
            "tolerances": self.tolerances.model_dump(mode="json"),
            **({"coarse_runs": self.coarse_runs} if self.coarse is not None else {}),
            **({"early_until": self.early_until, "early_runs": self.early_runs}
               if self.early_until is not None else {}),
        }  # fmt: skip


#: The ``forecast_done`` label of the surrogate's forecasts; every other one is the simulator's.
SURROGATE_LABEL = "surrogate"

#: The end of the ``forecast_done`` labels of a coarse-corrected model's coarse runs.
COARSE_LABEL = "(coarse)"

#: The end of the ``forecast_done`` labels of an early-start model's early runs.
EARLY_LABEL = "(early)"


def _side(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Sum one source's forecasts: wall seconds, members, and worker-seconds if known."""
    members = sum(int(r.get("n_ok", 0)) + int(r.get("n_failed", 0)) for r in records)
    seconds = sum(float(r.get("seconds", 0.0)) for r in records)
    # Worker-seconds: a forecast's wall time times the members it ran at once. Per member,
    # comparable with one full run's own time (the build's), however many ran beside it.
    known = all(isinstance(r.get("workers"), int) for r in records)
    worker_seconds = (
        sum(float(r.get("seconds", 0.0)) * int(r["workers"]) for r in records) if known else None
    )
    return {
        "forecasts": len(records),
        "members": members,
        "seconds": seconds,
        "seconds_per_member": seconds / members if members else None,
        "worker_seconds": worker_seconds,
        "worker_seconds_per_member": (
            worker_seconds / members if members and worker_seconds is not None else None
        ),
    }


def forecast_timing(events: Iterable[Mapping[str, Any]], cost: BuildCost | None) -> dict[str, Any]:
    """Return a surrogate run's forecast times against the full model's (decision 14).

    Parameters
    ----------
    events
        The run's structured events (``events.jsonl``, every session of it): the
        ``forecast_done`` ones are used, ``label = "surrogate"`` for the surrogate's,
        labels ending ``(coarse)`` for a coarse-corrected model's coarse runs and
        ``(early)`` for an early-start model's early runs,
        every other label the simulator's (checks, members outside the domain, the
        posterior's forecast, a switch to the simulator).
    cost
        The build's full runs (``None``: unknown, and no speed-up is given).

    Returns
    -------
    dict
        ``surrogate``, ``simulator``, ``coarse`` and ``early`` (the coarse or early runs,
        part of each prediction's cost): forecasts, members, wall ``seconds`` and per member,
        ``worker_seconds`` (wall times members at once) and per member; ``full_model``: the
        build's cost; ``speedup``: the full model's mean run time over the surrogate's
        worker-seconds per member (with its coarse run's, for a coarse-corrected model);
        ``simulator_equivalent_worker_seconds``: what the surrogate's members would have
        taken as full runs (the build's mean each), against ``run_worker_seconds`` (the
        surrogate's, the coarse runs' and the simulator's), the build's ``total_seconds`` a
        one-off cost beside them. Times are measured, not part of
        the run's results (never checkpointed).
    """
    done = [e for e in events if e.get("event") == "forecast_done"]
    surrogate = _side([e for e in done if e.get("label") == SURROGATE_LABEL])
    coarse = _side([e for e in done if str(e.get("label", "")).endswith(COARSE_LABEL)])
    early = _side([e for e in done if str(e.get("label", "")).endswith(EARLY_LABEL)])
    simulator = _side([
        e for e in done
        if e.get("label") != SURROGATE_LABEL
        and not str(e.get("label", "")).endswith((COARSE_LABEL, EARLY_LABEL))
    ])  # fmt: skip
    per_member = surrogate["worker_seconds_per_member"]
    for companion in (coarse, early):
        if per_member is not None and companion["members"]:
            # Each prediction costs its coarse or early run too.
            each = companion["worker_seconds_per_member"]
            per_member = None if each is None else per_member + each
    mean = None if cost is None else cost.mean_seconds
    both = (
        surrogate["worker_seconds"], simulator["worker_seconds"], coarse["worker_seconds"],
        early["worker_seconds"],
    )  # fmt: skip
    return {
        "surrogate": surrogate,
        "simulator": simulator,
        "coarse": coarse,
        "early": early,
        "full_model": None if cost is None else cost.model_dump(mode="json"),
        "speedup": mean / per_member if mean is not None and per_member else None,
        "simulator_equivalent_worker_seconds": (
            None if mean is None else surrogate["members"] * mean
        ),
        "run_seconds": (
            surrogate["seconds"] + simulator["seconds"] + coarse["seconds"] + early["seconds"]
        ),
        "run_worker_seconds": None if None in both else float(sum(both)),
    }


def model_keys_rows(model_keys: Sequence[DataKey], keys: Sequence[DataKey]) -> list[int]:
    """Return the row of each of ``model_keys`` in ``keys`` (all must be there)."""
    index = {key: row for row, key in enumerate(keys)}
    return [index[key] for key in model_keys]
