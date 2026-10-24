"""The ``ert`` simulator: an ERT case's forward model run by ERT for APEX.

APEX keeps its own loop and parameters; ERT runs the case's forward model steps (its
``FORWARD_MODEL`` lines, e.g. ``APEX_RENDER_DECK`` and ``FLOW``, or ``ECLIPSE100`` on a deck
that ``INCLUDE``s the fields). Members go to ERT in batches, one ``ert ensemble_experiment``
per forecast, by the ``ert`` executor (:mod:`apex.execution.ert`); this simulator says what
a batch needs:

- :meth:`write_parameters` writes each field of a member into its run path as the case's
  ``FIELD`` output file (``<keyword>`` in GRDECL, every value exact), where the forward
  model reads it: ERT itself never sees the parameters, so it neither samples, updates nor
  rounds them (ERT stores fields as float32);
- :meth:`forward_config` is the case's configuration without what ERT would own here
  (parameters, observations, run paths, queue, hooks), for the executor to complete;
- :meth:`read` reads a finished realization's summary, or :meth:`read_batch` a whole
  batch's responses of every observation type from its ERT storage.

:meth:`run` refuses: a member runs only in a batch, through the ``ert`` executor.
"""

import json
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import MappingProxyType
from typing import Annotated, ClassVar, Literal

import numpy as np
from numpy.typing import NDArray

from apex.core.errors import SimulationError
from apex.core.options import TEXT, InputFile, ProgramArguments, Text, mapping
from apex.core.plugins import simulators
from apex.core.simulation import MemberInput, MemberSeries, SeriesRequest
from apex.domain.observations import DataKey
from apex.simulation.deck import read_series, read_summary
from apex.simulation.grdecl import write_grdecl

#: Keywords of the case that ERT does not own in a batch: APEX supplies the parameters and
#: the observations, and the executor the realizations, run paths, storage and queue.
OWNED_BY_APEX = frozenset({
    "FIELD", "GEN_KW", "SURFACE", "DESIGN_MATRIX", "OBS_CONFIG", "GRID", "NUM_REALIZATIONS",
    "MIN_REALIZATIONS", "RUNPATH", "ENSPATH", "QUEUE_SYSTEM", "QUEUE_OPTION", "MAX_RUNNING",
    "MAX_SUBMIT", "HOOK_WORKFLOW", "LOAD_WORKFLOW", "LOAD_WORKFLOW_JOB", "ANALYSIS_SET_VAR",
    "ENKF_ALPHA", "STD_CUTOFF", "RANDOM_SEED", "UPDATE_LOG_PATH", "RUNPATH_FILE",
})  # fmt: skip

#: The file in a run path that names the APEX member ERT's realization runs (a deck's
#: ``member_id`` is APEX's, not ERT's realization number in the batch).
MEMBER_FILE = "APEX_MEMBER"

#: The file of a member's step parameters in its run path (read by the batch's parameter
#: step, :mod:`apex.integrations.ert.parameters`).
PARAMETERS_FILE = "apex_parameters.npz"

#: A field's keyword in the run path's GRDECL file and that file's name.
FIELD_ROLES = mapping(keyword=TEXT, file=TEXT)


@simulators.register("ert")
class ErtCase:
    """An ERT case's forward model, run by ERT through the ``ert`` executor.

    Parameters
    ----------
    case_file
        The ERT configuration (``.ert``).
    fields
        ``{state variable: {keyword, file}}``: each field's ERT name (its GRDECL keyword)
        and the output file the forward model reads in the run path.
    eclbase
        The case's ``ECLBASE``: the simulator's case name in the run path (``<IENS>``,
        ``<ITER>`` are replaced).
    responses
        ``"summary"`` (default): each realization's summary is read (:meth:`read`).
        ``"storage"``: the batch keeps the case's ``OBS_CONFIG``, and the responses of every
        observation type are read from the batch's ERT storage (:meth:`read_batch`).
    results_program
        For ``"storage"``: the program that reads them
        (``python -m apex.integrations.ert.observations``, set by the case's builder).
    step_parameters
        ``{state variable (or property): ERT name}`` of the parameters ERT's own writers
        put into the run path: ``GEN_KW`` groups, surfaces, and fields in other formats or
        with transforms. They are saved in ``apex_parameters.npz``.
    parameter_step
        The forward-model step that writes them (``APEX_WRITE_PARAMETERS(<CASE>=...)``),
        put before the case's first ``FORWARD_MODEL``.
    skip_forward_steps
        The case's first ``FORWARD_MODEL`` steps to leave out of every batch: those that
        make ``FORWARD_INIT`` parameters, run once for the prior (``ert_forward_init``).
    """

    #: ERT itself (the ``ert`` extra; checked by ``apex plugins``, not imported).
    requires: ClassVar[Mapping[str, str]] = MappingProxyType({"ert": "ert"})

    needs_work_dir = True
    #: Runs only in batches, through the ``ert`` executor.
    ert_batches = True

    def __init__(
        self,
        case_file: InputFile,
        fields: Annotated[Mapping[str, Mapping[str, str]], FIELD_ROLES],
        eclbase: Text,
        responses: Literal["summary", "storage"] = "summary",
        results_program: ProgramArguments = (),
        step_parameters: Annotated[Mapping[str, str] | None, TEXT] = None,
        parameter_step: Text | None = None,
        skip_forward_steps: int = 0,
    ) -> None:
        self.case_file = Path(case_file).resolve()
        if not self.case_file.is_file():
            raise ValueError(f"ERT case not found: {self.case_file}")
        self.step_parameters = dict(step_parameters or {})
        if not fields and not self.step_parameters:
            raise ValueError("the ert simulator needs at least one field or step parameter")
        if self.step_parameters and not parameter_step:
            raise ValueError("step_parameters need the parameter_step that writes them")
        self.parameter_step = parameter_step
        if skip_forward_steps < 0:
            raise ValueError("skip_forward_steps must be >= 0")
        self.skip_forward_steps = int(skip_forward_steps)
        self.fields = {
            variable: (str(field["keyword"]), str(field["file"]))
            for variable, field in fields.items()
        }
        self.eclbase = eclbase
        if responses not in ("summary", "storage"):
            raise ValueError(f"responses must be 'summary' or 'storage', got {responses!r}")
        if responses == "storage" and not results_program:
            raise ValueError("responses = 'storage' needs the results_program that reads them")
        self.responses = responses
        self.results_program = tuple(results_program)

    @property
    def resources(self) -> dict[str, int]:
        """One core per realization while ERT runs it."""
        return {"cpu": 1}

    def run(self, member: MemberInput, keys: Sequence[DataKey]) -> NDArray[np.float64]:
        """Refuse: ERT runs members in batches (the ``ert`` executor).

        Raises
        ------
        SimulationError
            Always.
        """
        raise SimulationError(
            "the 'ert' simulator runs members through ERT in batches: use the 'ert' executor"
        )

    def write_parameters(self, member: MemberInput) -> None:
        """Write the member's fields into its run path, as the forward model reads them.

        Raises
        ------
        SimulationError
            If the member has no run path or lacks a field.
        """
        if member.work_dir is None:
            raise SimulationError(f"member {member.member_id} has no run path")
        for variable, (keyword, file) in self.fields.items():
            if variable not in member.state:
                raise SimulationError(f"member {member.member_id} has no {variable}")
            write_grdecl(member.work_dir / file, keyword, np.asarray(member.state[variable]))
        if self.step_parameters:
            missing = sorted(set(self.step_parameters) - set(member.state))
            if missing:
                raise SimulationError(f"member {member.member_id} has no {', '.join(missing)}")
            # Full precision, by ERT name, for the batch's parameter step to write.
            np.savez(
                member.work_dir / PARAMETERS_FILE,
                **{ert: np.asarray(member.state[v]) for v, ert in self.step_parameters.items()},  # type: ignore[arg-type]
            )
        (member.work_dir / MEMBER_FILE).write_text(f"{member.member_id}\n")

    def case_name(self, realization: int, iteration: int = 0) -> str:
        """Return ``ECLBASE`` with ERT's magic strings replaced."""
        return self.eclbase.replace("<IENS>", str(realization)).replace("<ITER>", str(iteration))

    def read(
        self, run_path: Path, realization: int, keys: Sequence[DataKey], iteration: int = 0
    ) -> NDArray[np.float64]:
        """Read a finished realization's summary at ``keys``.

        Raises
        ------
        SimulationError
            If its summary is missing (the realization failed) or lacks a key.
        """
        return read_summary(run_path / self.case_name(realization, iteration), keys)

    def read_series(
        self, run_path: Path, realization: int, request: SeriesRequest, iteration: int = 0
    ) -> MemberSeries:
        """Read a finished realization's whole series (predictions).

        Raises
        ------
        SimulationError
            If its summary is missing (the realization failed) or lacks a vector.
        """
        return read_series(run_path / self.case_name(realization, iteration), request)

    def forward_config(self) -> list[str]:
        """Return the case's lines without what ERT does not own in a batch.

        Comments and continuation-free lines whose first word is in :data:`OWNED_BY_APEX`
        are dropped; everything else (``FORWARD_MODEL``, ``ECLBASE``, ``DATA_FILE``,
        ``DEFINE``, ``SETENV``, ``INSTALL_JOB``, ``INCLUDE``, ...) is kept as written.
        """
        # With responses read from the batch's storage, ERT matches them to the case's own
        # observations, so the batch keeps OBS_CONFIG.
        owned = OWNED_BY_APEX - {"OBS_CONFIG"} if self.responses == "storage" else OWNED_BY_APEX
        kept = []
        step_placed = not self.step_parameters
        skipped = 0
        for line in self.case_file.read_text().splitlines():
            words = line.split("--", 1)[0].split()
            if words and words[0].upper() in owned:
                continue
            if words and words[0].upper() == "FORWARD_MODEL" and skipped < self.skip_forward_steps:
                # The steps that make FORWARD_INIT parameters ran once, for the prior; APEX
                # writes those parameters itself in every batch.
                skipped += 1
                continue
            if not step_placed and words and words[0].upper() == "FORWARD_MODEL":
                # The parameters are written before the case's own steps read them.
                kept.append(f"FORWARD_MODEL {self.parameter_step}")
                step_placed = True
            kept.append(line)
        if not step_placed:
            kept.append(f"FORWARD_MODEL {self.parameter_step}")
        return kept

    def read_batch(
        self,
        storage: Path,
        keys: Sequence[DataKey],
        n_realizations: int,
        environment: Mapping[str, str],
    ) -> NDArray[np.float64]:
        """Read a finished batch's responses at ``keys`` from its ERT storage.

        Returns ``(n_keys, n_realizations)``, NaN where a realization has no response.
        Runs ``results_program`` (ERT's own matching of responses to observations).

        Raises
        ------
        SimulationError
            If the program fails.
        """
        work = storage.parent
        (work / "keys.json").write_text(json.dumps([[k.time, k.datatype] for k in keys]))
        out = work / "responses.npy"
        done = subprocess.run(
            [*self.results_program, "--storage", str(storage), "--keys", str(work / "keys.json"),
             "--realizations", str(n_realizations), "--out", str(out)],
            env=dict(environment), capture_output=True, text=True, check=False,
        )  # fmt: skip
        if done.returncode != 0:
            raise SimulationError(f"reading the batch's responses failed: {done.stderr[-2000:]}")
        values: NDArray[np.float64] = np.load(out)
        return values
