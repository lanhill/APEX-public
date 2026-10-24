"""The forward-model step ``APEX_WRITE_PARAMETERS``.

In ``APEX_HISTORY_MATCH`` batches ERT does not own the parameters: the ``ert`` simulator
writes each member's values, at full precision, into its run path. ASCII GRDECL fields
without transforms it writes itself; everything else goes into ``apex_parameters.npz``,
which this step, the batch's first forward-model step, writes as ERT itself would:

- **GEN_KW groups:** the latent (standard-normal) values, transformed by ERT's own
  distribution of each key, into ``parameters.json`` and ``parameters.txt`` (with ERT's
  ``LOG10_`` entries) and the case's templates, through ERT's own writers and
  substitution.
- **SURFACE:** IRAP (ASCII or binary) with the surface's geometry, as ERT writes it.
- **FIELD** in ROFF or binary GRDECL, or with an ``OUTPUT_TRANSFORM``: ERT's transform
  and ``MIN``/``MAX`` truncation, then ERT's ``save_field`` in the field's format.

The array of each parameter is named by its ERT name (a ``GEN_KW`` group's name, with
its keys' latent values in the case's order; a surface's or field's name, its values
flat, Eclipse order for fields).

    python -m apex.integrations.ert.parameters --case CASE.ert --realization 3
"""

import argparse
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

#: The file the ert simulator writes into each run path.
PARAMETERS_FILE = "apex_parameters.npz"


def gen_kw_keys(config: Any, group: str) -> list[Any]:  # noqa: ANN401 (ERT's configs)
    """Return the ``GenKwConfig`` of each key of ``group``, in the case's order."""
    from ert.config import GenKwConfig

    return [
        p for p in config.ensemble_config.parameter_configs.values()
        if isinstance(p, GenKwConfig) and p.group_name == group
    ]  # fmt: skip


def write_parameters(
    config: Any,  # noqa: ANN401 (a parsed ErtConfig)
    run_path: Path,
    realization: int,
    values: dict[str, NDArray[np.float64]],
    iteration: int = 0,
) -> None:
    """Write ``values`` (by ERT parameter name) into ``run_path`` as ERT would.

    Raises
    ------
    ValueError
        If a name is not a parameter of the case, or its values have the wrong size.
    """
    import ert.run_models._create_runpath as run_path_files
    from ert.config import Field, SurfaceConfig
    from ert.config.distribution import LogNormalSettings, LogUnifSettings
    from ert.field_utils import save_field
    from ert.substitutions import Substitutions
    from surfio import IrapHeader, IrapSurface

    parameters = config.ensemble_config.parameter_configs
    exports: dict[str, dict[str, float | str]] = {}
    log_exports: dict[str, dict[str, float | str]] = {}
    for name, array in values.items():
        keys = gen_kw_keys(config, name)
        if keys:
            if array.size != len(keys):
                raise ValueError(f"GEN_KW {name} has {len(keys)} keys, not {array.size} values")
            for key, latent in zip(keys, np.asarray(array, dtype=np.float64), strict=True):
                value = float(key.transform_numpy(np.array([latent]))[0])
                exports.setdefault(name, {})[key.name] = value
                if isinstance(key.distribution, (LogNormalSettings, LogUnifSettings)) and value > 0:
                    log_exports.setdefault(f"LOG10_{name}", {})[key.name] = math.log10(value)
            continue
        parameter = parameters.get(name)
        if isinstance(parameter, SurfaceConfig):
            grid = np.asarray(array, dtype=np.float32).reshape(parameter.ncol, parameter.nrow)
            surface = IrapSurface(
                header=IrapHeader(
                    ncol=parameter.ncol, nrow=parameter.nrow, xori=parameter.xori,
                    yori=parameter.yori, xinc=parameter.xinc,
                    yinc=parameter.yinc * parameter.yflip, rot=parameter.rotation,
                    xrot=parameter.xori, yrot=parameter.yori,
                ),
                values=grid,
            )  # fmt: skip
            target = run_path / _substituted(str(parameter.output_file), realization, iteration)
            target.parent.mkdir(parents=True, exist_ok=True)
            if parameter.file_format == "binary":
                surface.to_binary_file(target)
            else:
                surface.to_ascii_file(target)
        elif isinstance(parameter, Field):
            shape = (parameter.nx, parameter.ny, parameter.nz)
            if array.size != math.prod(shape):
                raise ValueError(f"FIELD {name} has {math.prod(shape)} cells, not {array.size}")
            # APEX's fields are flat in Eclipse order (i fastest); ERT's are (nx, ny, nz).
            cube = np.asarray(array, dtype=np.float64).reshape(shape, order="F")
            target = run_path / _substituted(str(parameter.output_file), realization, iteration)
            save_field(_output(parameter, cube), parameter.name, target, parameter.file_format)
        else:
            raise ValueError(f"{name} is not a GEN_KW group, SURFACE or FIELD of the case")
    if exports:
        export_name = config.runpath_config.gen_kw_export_name
        run_path_files._value_export_txt(run_path, export_name, exports | log_exports)
        run_path_files._value_export_json(run_path, export_name, exports)
    substituter = Substitutions(config.substitutions).real_iter_substituter(realization, iteration)
    with_parameters = run_path_files._make_param_substituter(substituter, exports)
    for template, target_name in config.ert_templates:
        target = run_path / substituter.substitute(target_name)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(with_parameters.substitute(Path(template).read_text()))


def _output(parameter: Any, cube: NDArray[np.float64]) -> np.ma.MaskedArray[Any, Any]:  # noqa: ANN401
    """Return a field's output values as ERT computes them: transform, then truncate."""
    from ert.config.field import _field_truncate, field_transform

    transformed = field_transform(cube.astype(np.float32), parameter.output_transformation)
    truncated = _field_truncate(transformed, parameter.truncation_min, parameter.truncation_max)
    return np.ma.MaskedArray(truncated, mask=np.zeros(cube.shape, dtype=bool))


def _substituted(name: str, realization: int, iteration: int) -> str:
    return name.replace("<IENS>", str(realization)).replace("<ITER>", str(iteration))


def main(argv: list[str] | None = None) -> int:
    """Run the step in the current run path (see the module docstring)."""
    from apex.integrations.ert.config import parse_case

    parser = argparse.ArgumentParser(description="Write APEX's parameters as ERT would.")
    parser.add_argument("--case", type=Path, required=True)
    parser.add_argument("--realization", type=int, required=True)
    parser.add_argument("--iteration", type=int, default=0)
    args = parser.parse_args(argv)
    run_path = Path.cwd()
    with np.load(run_path / PARAMETERS_FILE) as stored:
        values = {name: stored[name] for name in stored.files}
    config = parse_case(args.case)
    write_parameters(config, run_path, args.realization, values, args.iteration)
    return 0


if __name__ == "__main__":
    sys.exit(main())
