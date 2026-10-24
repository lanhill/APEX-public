"""``FORWARD_INIT:True`` parameters for ``APEX_HISTORY_MATCH``.

In ERT, a parameter with ``FORWARD_INIT:True`` is made by the forward model at iteration 0
(e.g. by a geomodelling job) and read back from the run path afterwards. APEX draws its
prior before its first forecast, so the joint prior generator ``ert_forward_init`` runs
those steps first: one ``ert ensemble_experiment`` of the case with only its first
``steps`` ``FORWARD_MODEL`` lines, no responses and no observations, and then reads each
realization's parameters from its run path with ERT's own readers (fields: ERT's
``read_field`` and ``INIT_TRANSFORM``, flat in Eclipse order; surfaces: IRAP).

The run then goes on as any other: the first forecast runs the whole forward model on
this prior, which the ``ert`` simulator writes back into each run path. As in ERT, the
forward model must not remake the parameters after iteration 0.
"""

import shutil
import subprocess
from collections.abc import Sequence
from pathlib import Path
from typing import Annotated

import numpy as np
from numpy.typing import NDArray

from apex.core.options import OUTPUT, InputFile, Program
from apex.core.plugins import priors
from apex.domain.state import StateVariable
from apex.execution.ert import ert_environment
from apex.simulation.ert import OWNED_BY_APEX

#: Keywords an initialization batch leaves out besides those APEX owns: the responses,
#: which the shortened forward model does not make.
RESPONSES = frozenset({"SUMMARY", "GEN_DATA", "RFT", "BREAKTHROUGH", "SEISMIC"})


@priors.register("ert_forward_init")
class ErtForwardInit:
    """Draw ``FORWARD_INIT:True`` parameters by running the case's generating steps.

    Parameters
    ----------
    case_file
        The ERT case.
    steps
        How many of the case's first ``FORWARD_MODEL`` steps make the parameters.
    work_dir
        Where the initialization batch runs (kept, for inspection).
    workers
        Realizations ERT runs at once.
    ert_executable
        The ``ert`` program; default the one in APEX's environment.
    """

    #: The forward model's values are used as they are.
    clip = False

    def __init__(
        self,
        case_file: InputFile,
        steps: int,
        work_dir: Annotated[str, OUTPUT],
        workers: int = 1,
        ert_executable: Program | None = None,
    ) -> None:
        if steps < 1:
            raise ValueError("steps must be at least 1")
        self.case_file = Path(case_file).resolve()
        self.steps = int(steps)
        self.work_dir = Path(work_dir)
        self.workers = int(workers)
        self.ert_executable = ert_executable

    def generate_joint(
        self,
        variables: Sequence[StateVariable],
        n_ensemble: int | None,
        rng: np.random.Generator,
    ) -> dict[str, NDArray[np.float64]]:
        """Run the generating steps and read every realization's parameters.

        Returns ``{variable: (size, n_ensemble)}``. ``rng`` is not used: the forward model
        makes the parameters (its own randomness, e.g. seeded by ``<IENS>``, is the
        case's).

        Raises
        ------
        ValueError
            If ``n_ensemble`` is missing, the batch fails, or a realization's file is
            missing or of the wrong size.
        """
        if n_ensemble is None:
            raise ValueError("ert_forward_init needs [prior] ensemble_size")
        from ert.config import Field

        from apex.integrations.ert.config import parse_case

        case = parse_case(self.case_file)
        by_name = {name.lower(): p for name, p in case.ensemble_config.parameter_configs.items()}
        unknown = [v.name for v in variables if v.name not in by_name]
        if unknown:
            raise ValueError(f"no parameter of the case is called {', '.join(unknown)}")
        runs = self._run(n_ensemble)
        drawn = {}
        for variable in variables:
            parameter = by_name[variable.name]
            columns = []
            for i in range(n_ensemble):
                name = parameter.forward_init_file.replace("<IENS>", str(i))
                path = runs / f"realization-{i}" / name.replace("<ITER>", "0").replace("%d", str(i))
                columns.append(_read(parameter, path, isinstance(parameter, Field)))
            block = np.ascontiguousarray(np.stack(columns, axis=-1))
            if block.shape[0] != variable.size:
                raise ValueError(f"{variable.name}: {block.shape[0]} values, not {variable.size}")
            drawn[variable.name] = block
        return drawn

    def _run(self, n_ensemble: int) -> Path:
        """Run the initialization batch; return the folder of its run paths."""
        if self.work_dir.exists():
            shutil.rmtree(self.work_dir)
        runs = self.work_dir / "runs"
        runs.mkdir(parents=True)
        lines, forward_steps = [], 0
        for line in self.case_file.read_text().splitlines():
            words = line.split("--", 1)[0].split()
            keyword = words[0].upper() if words else ""
            if keyword in OWNED_BY_APEX or keyword in RESPONSES:
                continue
            if keyword == "FORWARD_MODEL":
                forward_steps += 1
                if forward_steps > self.steps:
                    continue
            lines.append(line)
        if forward_steps < self.steps:
            raise ValueError(f"the case has {forward_steps} FORWARD_MODEL steps, not {self.steps}")
        config = self.case_file.parent / f".apex_forward_init_{self.work_dir.name}.ert"
        config.write_text("\n".join([
            *lines,
            "",
            "-- APEX's initialization batch: the steps that make FORWARD_INIT.",
            f"NUM_REALIZATIONS {n_ensemble}",
            f"RUNPATH {runs}/realization-<IENS>",
            f"ENSPATH {self.work_dir / 'storage'}",
            f"RUNPATH_FILE {self.work_dir / 'runpath_list'}",
            "QUEUE_SYSTEM LOCAL",
            f"QUEUE_OPTION LOCAL MAX_RUNNING {self.workers}",
            "MIN_REALIZATIONS 1",
        ]) + "\n")  # fmt: skip
        environment = ert_environment()
        executable = self.ert_executable or shutil.which("ert", path=environment["PATH"])
        if executable is None:
            raise ValueError("`ert` is not installed: install apex[ert] (ERT 26.x)")
        try:
            with (self.work_dir / "ert.log").open("w") as log:
                done = subprocess.run(
                    [executable, "--logdir", str(self.work_dir / "logs"), "ensemble_experiment",
                     config.name],
                    cwd=config.parent, env=environment, stdout=log, stderr=subprocess.STDOUT,
                    check=False,
                )  # fmt: skip
        finally:
            config.unlink(missing_ok=True)
        if done.returncode != 0:
            raise ValueError(f"the FORWARD_INIT batch failed (see {self.work_dir / 'ert.log'})")
        return runs


def _read(parameter: object, path: Path, is_field: bool) -> NDArray[np.float64]:
    """Read one realization's parameter as ERT stores it."""
    from ert.config.field import field_transform
    from ert.field_utils import read_field
    from surfio import IrapSurface

    if not path.exists():
        raise ValueError(f"{path} is missing: the FORWARD_INIT steps did not make it")
    if is_field:
        shape = (parameter.nx, parameter.ny, parameter.nz)  # type: ignore[attr-defined]
        values = read_field(path, parameter.name, shape)  # type: ignore[attr-defined]
        stored = field_transform(values, parameter.input_transformation)  # type: ignore[attr-defined]
        return np.asarray(stored, dtype=np.float64).ravel(order="F")
    surface = (
        IrapSurface.from_binary_file(path)
        if parameter.file_format == "binary"  # type: ignore[attr-defined]
        else IrapSurface.from_ascii_file(path)
    )
    return np.asarray(surface.values, dtype=np.float64).ravel()
