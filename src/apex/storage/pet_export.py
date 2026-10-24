"""Convert an APEX history-matching case into a PET case that PET can run.

The inverse of :mod:`apex.storage.pet_config`, from the importer's own tables read in
reverse: what this module writes, ``apex pet-import`` reads back to the same APEX
configuration, and PET (upstream, at the reference commit) reads without a problem
(``input_output.config.validate``; the opt-in test runs it in ``.venv-pet``). Written, in
PET's own form (as in PET's own tutorials):

- ``pet.toml``: ``[ensemble]`` (``ne``, ``state``, ``importstate`` or each variable's
  ``prior_<name>`` variogram, ``limits``), ``[dataassim]`` (``scheme``, ``analysis``,
  ``energy`` in percent, the MDA or iteration block, ``obsname``, ``data``, ``datavar``)
  and ``[simulator]`` (``runfile``, ``datatype``, ``reporttype``, ``reportpoint`` as dates,
  ``parallel``, ``sim_limit``);
- ``data.csv`` and ``var.csv``: PET's wide tables, one row per report date, each variance an
  ``['abs', v]`` cell;
- ``RUNFILE.mako``: APEX's template with ``${case_dir}/`` as ``../`` (PET renders it in
  ``<case>/En_<i>/``), and the files it includes, copied;
- the prior ensemble (``importstate``) and a variogram prior's mean file.

Everything is checked before anything is written. What PET cannot run as APEX does is
refused with the reason, never approximated: data transforms, geophysics,
parameterizations, model error, cross validation, field data, localization, damping, the
methods and strategies PET lacks, method options it has no place for, GIES (PET-fork's
own configuration format is not written yet), template values, one-sided bounds, joint
priors, and variogram priors PET's sampler would not reproduce (any but the isotropic or
j-anisotropic, unrotated fields of a square grid). What differs although it runs (the
stopping rules of the EnRML methods) is noted.
"""

import csv
import datetime as dt
import io
import re
import shutil
import tomllib
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import tomli_w

from apex.core.errors import ConfigError
from apex.storage.pet_config import _GN_KEYS, _LM_KEYS, _PET_VARIOGRAMS, _pet_state_scaling

#: APEX methods PET runs (upstream), and PET's scheme for each.
SCHEMES = {"es": "es", "esmda": "esmda", "lm_enrml": "lmenrml", "gn_enrml": "gnenrml"}
#: The analyses each scheme has in PET (the importer's ``_ENRML_SCHEMES``).
ANALYSES = {
    "es": ("approx",),
    "esmda": ("approx",),
    "lm_enrml": ("approx", "full", "subspace", "subspace2"),
    "gn_enrml": ("approx", "full", "subspace", "subspace2", "margis"),
}
#: The method options PET has a place for, per method.
METHOD_OPTIONS: dict[str, set[str]] = {"es": set(), "esmda": {"inflation"},
                                       "lm_enrml": {"max_inner_iterations"},
                                       "gn_enrml": {"max_inner_iterations"}}  # fmt: skip
#: The method keys this module reads; any other is refused.
METHOD_KEYS = {"name", "options", "strategy", "schedule", "max_iterations", "diagnostics"}
#: Simulator options PET's runfile carries (the others are PET's simulator's own: noted).
SIMULATOR_OPTIONS = {"deck_template_file", "case_dir", "timeout_s", "template_values"}
#: Iteration keys, APEX's schedule key to PET's: the importer's tables read in reverse.
LM_KEYS = {apex: pet for pet, apex in _LM_KEYS} | {"data_mismatch_tolerance": "data_misfit_tol"}
GN_KEYS = {apex: pet for pet, apex in _GN_KEYS} | {"data_mismatch_tolerance": "data_misfit_tol"}
#: APEX variogram models to PET's (the importer's table read in reverse).
VARIOGRAMS = {model: pet for pet, model in _PET_VARIOGRAMS.items()}
#: APEX's truncation energy where a strategy's options leave it out (ApproxUpdate,
#: FullUpdate, SubspaceUpdate): written, since PET's EnRML default is 0.95.
APEX_ENERGY = 0.98
#: What LM-EnRML and GN-EnRML use with no [method.strategy] at all (ApproxUpdate(0.95)).
APEX_ENRML_ENERGY = 0.95
#: Strategies with no truncation (no energy).
TRANSFORMS = ("subspace2", "margis")
#: A gaussian_field's options this module writes in PET's terms; any other is refused.
FIELD_OPTIONS = {"grid", "mean", "mean_file", "variance", "major_range", "minor_range",
                 "azimuth", "variogram", "exponent"}  # fmt: skip
REFUSED_SECTIONS = {
    "data_transforms": "data transforms (e.g. wavelet compression) are not exported yet",
    "geophysics": "geophysical data are not exported",
    "parameterizations": "PET has no parameterizations",
    "model_error": "PET has no model-error characterization",
    "validation": "PET has no cross validation",
}
#: Sections that are APEX's own: left out, with a note.
APEX_ONLY = ("metrics", "output", "execution", "case", "notifications")
#: SimulatorWrap upper-cases the runfile's name: the file must be upper case.
RUNFILE = "RUNFILE"


@dataclass
class ExportResult:
    """The PET configuration written, and notes on what was left out or differs."""

    path: Path
    notes: list[str] = field(default_factory=list)


@dataclass
class _Plan:
    """What will be written: text files and copies, once everything is checked."""

    texts: dict[str, str] = field(default_factory=dict)
    copies: dict[str, Path] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


def export_pet_case(config_path: Path, output_dir: Path) -> ExportResult:
    """Write a PET case for the APEX configuration at ``config_path`` (see the module).

    Raises
    ------
    ConfigError
        If the configuration uses what PET cannot run, or a file it names is missing or
        malformed. Nothing is written then.
    """
    try:
        return _export(config_path, output_dir)
    except ConfigError:
        raise
    except (KeyError, ValueError, TypeError, OSError) as error:
        # The error's own words, with paths as file names (they reach users and notes).
        reason = re.sub(r"(?:/[\w.@+-]+)+/([\w.@+-]+)", r"\1", str(error))
        raise ConfigError(f"the APEX configuration cannot be exported: {reason}") from error


def _export(config_path: Path, output_dir: Path) -> ExportResult:
    try:
        raw = tomllib.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise ConfigError(f"cannot read the APEX configuration {config_path}: {error}") from error
    base = config_path.resolve().parent
    plan = _Plan()
    refused = [f"[{key}]: {why}" for key, why in REFUSED_SECTIONS.items() if raw.get(key)]
    if refused:
        raise ConfigError("PET cannot run this case: " + "; ".join(refused))
    left_out = [key for key in APEX_ONLY if key in raw]
    if left_out:
        plan.notes.append(f"APEX's own sections are left out: {', '.join(left_out)}")
    ensemble = _ensemble(raw, base, plan)
    dataassim = _dataassim(dict(raw.get("method", {})), ensemble, raw, base, plan)
    times, columns = _observations(dict(raw.get("observations", {})), base, plan)
    simulator = _simulator(dict(raw.get("simulator", {})), dict(raw.get("execution", {})),
                           base, plan)  # fmt: skip
    dataassim.update({"obsname": "dates", "data": "data.csv", "datavar": "var.csv"})
    simulator.update({"datatype": columns, "reporttype": "dates", "reportpoint": times})
    if "seed" in dict(raw.get("run", {})):
        ensemble["seed"] = int(raw["run"]["seed"])
    pet = {"ensemble": ensemble, "dataassim": dataassim, "simulator": simulator}
    header = (
        f"# Converted from the APEX configuration {config_path.name} by APEX's case-converter.\n"
    )
    plan.texts["pet.toml"] = header + tomli_w.dumps(pet)
    # Checked: now written.
    output_dir.mkdir(parents=True, exist_ok=True)
    for relative, source in plan.copies.items():
        (output_dir / relative).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, output_dir / relative)
    for relative, text in plan.texts.items():
        (output_dir / relative).write_text(text)
    return ExportResult(output_dir / "pet.toml", plan.notes)


def _ensemble(raw: Mapping[str, Any], base: Path, plan: _Plan) -> dict[str, Any]:
    variables = list(dict(raw.get("state", {})).get("variables", []))
    if not variables:
        raise ConfigError("the case has no [[state.variables]]")
    prior = dict(raw.get("prior", {}))
    if prior.get("groups"):
        raise ConfigError("PET has no joint priors ([prior.groups])")
    names = [str(v["name"]) for v in variables]
    ensemble: dict[str, Any] = {"state": names}
    if "ensemble_size" in prior:
        ensemble["ne"] = int(prior["ensemble_size"])
    generators = {name: dict(prior.get("variables", {}).get(name, prior.get("generator", {})))
                  for name in names}  # fmt: skip
    kinds = {str(g.get("name", "gaussian")) for g in generators.values()}
    if kinds == {"file"}:
        options = [dict(g.get("options", {})) for g in generators.values()]
        files = {str(o.get("ensemble_file")) for o in options}
        if len(files) != 1:
            raise ConfigError("PET reads one stored ensemble (importstate) for every variable")
        if any(o.get("key") not in (None, name) for o, name in zip(options, names, strict=True)):
            raise ConfigError("PET reads each variable's array by its name: the stored "
                              "ensemble's `key` option has no place in PET")  # fmt: skip
        source = base / files.pop()
        if not source.is_file():
            raise ConfigError(f"prior ensemble not found: {source.name}")
        with np.load(source) as data:
            missing = [n for n in names if n not in data.files]
        if missing:
            raise ConfigError(f"{source.name} has no array named {', '.join(missing)}: PET "
                              "reads each variable's array by its name")  # fmt: skip
        plan.copies["prior_ensemble.npz"] = source
        ensemble["importstate"] = "prior_ensemble.npz"
    elif kinds == {"gaussian_field"}:
        if "ne" not in ensemble:
            raise ConfigError("a sampled prior needs [prior] ensemble_size")
        for name in names:
            field_options = dict(generators[name].get("options", {}))
            ensemble[f"prior_{name}"] = _variogram(name, field_options, base, plan)
    else:
        raise ConfigError(
            f"PET's priors are a stored ensemble or a variogram; this case uses "
            f"{', '.join(sorted(kinds))} (give a stored ensemble instead)"
        )
    for variable in variables:
        lower, upper = variable.get("lower"), variable.get("upper")
        if lower is None and upper is None:
            continue
        if lower is None or upper is None:
            raise ConfigError(f"{variable['name']}: PET's limits need both bounds")
        ensemble.setdefault(f"prior_{variable['name']}", {})["limits"] = [lower, upper]
    return ensemble


def _variogram(name: str, options: Mapping[str, Any], base: Path, plan: _Plan) -> dict[str, Any]:
    """Return ``prior_<name>`` for a ``gaussian_field`` PET's sampler reproduces."""
    unknown = sorted(set(options) - FIELD_OPTIONS)
    if unknown:
        raise ConfigError(f"prior of {name}: PET has no place for {unknown}")
    grid = [int(n) for n in options["grid"]]
    nz = grid[2] if len(grid) > 2 else 1

    def layers(key: str, default: object, cast: Callable[[Any], Any] = float) -> list[Any]:
        value = options.get(key, default)
        values = list(value) if isinstance(value, list) else [value] * nz
        if len(values) != nz:
            raise ConfigError(f"prior of {name}: {key} has {len(values)} entries for {nz} layers")
        return [cast(v) for v in values]

    varios = []
    models = layers("variogram", "spherical", str)
    for model, exponent in zip(models, layers("exponent", 1.0), strict=True):
        found = VARIOGRAMS.get((model, exponent))
        if found is None:
            raise ConfigError(f"prior of {name}: PET has no variogram {model!r} (exponent "
                              f"{exponent:g}); it has spherical, cubic and its exp")  # fmt: skip
        varios.append(found)
    major = layers("major_range", 10.0)
    minor = layers("minor_range", None) if options.get("minor_range") is not None else major
    azimuth = layers("azimuth", 0.0)
    aniso = [a / b for a, b in zip(major, minor, strict=True)]
    if grid[0] != grid[1]:
        raise ConfigError(f"prior of {name}: on a {grid[0]} x {grid[1]} grid PET's sampler "
                          "scrambles its field (docs/migration/pet_parity.md); give a stored "
                          "ensemble instead")  # fmt: skip
    # On a square grid PET's sampler transposes the field and does not rotate it: only
    # isotropic fields and fields anisotropic along j (azimuth 90) are ones it makes.
    for layer_aniso, layer_azimuth in zip(aniso, azimuth, strict=True):
        if not (layer_aniso == 1.0 and layer_azimuth == 0.0) and not (
            layer_aniso != 1.0 and layer_azimuth == 90.0
        ):
            raise ConfigError(
                f"prior of {name}: on a square grid PET's sampler makes isotropic fields, or "
                f"anisotropic ones along j (azimuth 90), unrotated; this one has azimuth "
                f"{layer_azimuth:g} with anisotropy {layer_aniso:g}"
            )
    table: dict[str, Any] = {"vario": varios, "grid": grid, "var": layers("variance", 1.0),
                             "range": minor, "aniso": aniso, "angle": [0.0] * nz}  # fmt: skip
    if "mean_file" in options:
        source = base / str(options["mean_file"])
        if source.suffix != ".npz" or not source.is_file():
            raise ConfigError(f"prior of {name}: PET reads its mean from an .npz file ("
                              f"{source.name})")  # fmt: skip
        with np.load(source) as data:
            if len(data.files) != 1:
                raise ConfigError(f"prior of {name}: PET reads a mean file of one array")
        target = f"{name}_prior_mean.npz"
        plan.copies[target] = source
        table["mean"] = target
    else:
        table["mean"] = options.get("mean", 0.0)
    return table


def _dataassim(
    method: Mapping[str, Any],
    ensemble: Mapping[str, Any],
    raw: Mapping[str, Any],
    base: Path,
    plan: _Plan,
) -> dict[str, Any]:
    name = str(method.get("name", ""))
    if name == "gies":
        raise ConfigError("GIES runs in PET-fork, whose own configuration format is not "
                          "exported yet")  # fmt: skip
    if name not in SCHEMES:
        raise ConfigError(f"PET has no method {name!r} (it has es, esmda, lm_enrml, gn_enrml)")
    extra_keys = sorted(set(method) - METHOD_KEYS)
    if extra_keys:
        raise ConfigError(f"PET has no place for [method] {', '.join(extra_keys)}")
    if method.get("diagnostics"):
        plan.notes.append("[method.diagnostics] is APEX's own: left out")
    options = dict(method.get("options", {}))
    extra_options = sorted(set(options) - METHOD_OPTIONS[name])
    if extra_options:
        raise ConfigError(f"PET's {name} has no option {extra_options}")
    strategy = dict(method.get("strategy", {}))
    analysis = str(strategy.get("name", "approx"))
    if analysis not in ANALYSES[name]:
        raise ConfigError(f"PET's {name} has the {', '.join(ANALYSES[name])} analyses, not "
                          f"{analysis!r}")  # fmt: skip
    strategy_options = dict(strategy.get("options", {}))
    scaling = strategy_options.pop("state_scaling_file", None)
    if analysis == "full":
        # PET's full analysis always scales by the prior's configured variances; APEX's,
        # without a scaling file, by the prior ensemble's spread: only PET's own exports.
        if scaling is None:
            raise ConfigError(
                "PET's full analysis scales by the prior's configured variances; this case "
                "scales by the prior ensemble (no state_scaling_file)"
            )
        _check_scaling(base / str(scaling), ensemble, raw, plan)
    elif scaling is not None:
        raise ConfigError(f"PET's {analysis} analysis has no state scaling")
    extra = sorted(set(strategy_options) - {"energy"})
    if extra:
        raise ConfigError(f"PET's {analysis} analysis has no option {extra}")
    default = APEX_ENRML_ENERGY if not strategy and name in ("lm_enrml", "gn_enrml") else (
        APEX_ENERGY)  # fmt: skip
    energy = None if analysis in TRANSFORMS else 100.0 * float(
        strategy_options.get("energy", default))  # fmt: skip
    found: dict[str, Any] = {"scheme": SCHEMES[name], "analysis": analysis}
    schedule = dict(method.get("schedule", {}))
    if name in ("es", "esmda"):
        if schedule:
            raise ConfigError(f"{name} has no schedule in PET")
        found["energy"] = energy
        if name == "esmda":
            inflation = [float(a) for a in options.get("inflation", [])]
            if not inflation:
                raise ConfigError("ES-MDA needs [method.options] inflation")
            found["mda"] = {"tot_assim_steps": len(inflation), "inflation_param": inflation}
        return found
    keys = LM_KEYS if name == "lm_enrml" else GN_KEYS
    unknown = sorted(set(schedule) - set(keys))
    if unknown:
        raise ConfigError(f"PET's {name} has no schedule option {unknown}")
    iteration: dict[str, Any] = {keys[k]: v if v == "auto" else float(v)
                                 for k, v in schedule.items()}  # fmt: skip
    iteration["max_iter"] = int(method.get("max_iterations", 10))
    if "max_inner_iterations" in options:
        iteration["max_inner_iter"] = int(options["max_inner_iterations"])
    if energy is not None:
        # PET's IterativeEnRML reads the energy from the iteration block only.
        iteration["energy"] = energy
    found["iteration"] = iteration
    plan.notes.append(
        f"{name}: APEX stops after two improving trials in a row and only on an improving "
        "one; PET stops on its first small improvement, or on a rejected trial, so the two "
        "may run different numbers of iterations"
    )
    return found


def _check_scaling(path: Path, ensemble: Mapping[str, Any], raw: Mapping[str, Any],
                   plan: _Plan) -> None:  # fmt: skip
    """Accept the full analysis' state scaling only if it is PET's, from the variances."""
    variables = list(dict(raw.get("state", {})).get("variables", []))
    expected = _pet_state_scaling(ensemble, variables, [])
    if expected is None or not path.is_file():
        raise ConfigError("the full analysis' state_scaling_file has no place in PET, which "
                          "scales by the prior's configured variances")  # fmt: skip
    with np.load(path) as data:
        given = np.asarray(data[data.files[0]], dtype=float)
    if given.shape != expected.shape or not np.allclose(given, expected):
        raise ConfigError("the full analysis' state_scaling_file is not PET's (from the prior's "
                          "variances): PET cannot run it")  # fmt: skip
    plan.notes.append("the full analysis' state scaling is PET's own, from the variances")


def _observations(
    observations: Mapping[str, Any], base: Path, plan: _Plan
) -> tuple[list[dt.datetime], list[str]]:
    """Plan PET's wide ``data.csv`` and ``var.csv``; return the report dates and data types."""
    if observations.get("fields"):
        raise ConfigError("field observations are not exported to PET yet")
    if "file" not in observations:
        raise ConfigError("the case has no [observations] file")
    path = base / str(observations["file"])
    if not path.is_file():
        raise ConfigError(f"observations not found: {path.name}")
    rows: dict[str, dict[str, tuple[str, str]]] = {}
    columns: list[str] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for number, row in enumerate(csv.DictReader(handle), start=2):
            time, datatype = row["time"].strip(), row["datatype"].strip()
            if datatype in rows.get(time, {}):
                raise ConfigError(f"{path.name}, line {number}: {datatype} at {time} twice")
            if datatype not in columns:
                columns.append(datatype)
            rows.setdefault(time, {})[datatype] = (row["value"].strip(), row["variance"].strip())
    dates = []
    for time in rows:
        try:
            dates.append(dt.datetime.fromisoformat(time))
        except ValueError:
            raise ConfigError(f"PET reports at dates; the observation time {time!r} is not "
                              "one") from None  # fmt: skip
    data, var = io.StringIO(), io.StringIO()
    data_writer = csv.writer(data, lineterminator="\n")
    var_writer = csv.writer(var, lineterminator="\n")
    data_writer.writerow(["dates", *columns])
    var_writer.writerow(["dates", *columns])
    for time, cells in rows.items():
        data_writer.writerow([time, *(cells[c][0] if c in cells else "" for c in columns)])
        var_writer.writerow([time, *(f"['abs', {cells[c][1]}]" if c in cells else ""
                                     for c in columns)])  # fmt: skip
    plan.texts["data.csv"] = data.getvalue()
    plan.texts["var.csv"] = var.getvalue()
    return dates, columns


#: ``${case_dir}/<path>`` in a template (as ``simulation.deck``'s, which storage may not
#: import): the files a template includes from its case folder.
_CASE_PATH = re.compile(r"\$\{case_dir\}/([^'\"\s,{}]+)")


def case_dir_files(template: Path) -> list[str] | None:
    """Return the paths ``template`` names under ``${case_dir}``.

    ``None`` if one is computed (``${case_dir}/grid_${level}.inc``) or leaves the folder.
    """
    found = [m.rstrip("/") for m in _CASE_PATH.findall(template.read_text(errors="replace"))]
    if any("$" in path or ".." in Path(path).parts for path in found):
        return None
    return sorted(set(found))


#: Mako expressions and control lines that name the member (PET's runfile has none).
_MEMBER = re.compile(r"\$\{[^}]*\bmember_id\b|^\s*%.*\bmember_id\b", re.MULTILINE)


def _simulator(
    simulator: Mapping[str, Any], execution: Mapping[str, Any], base: Path, plan: _Plan
) -> dict[str, Any]:
    if simulator.get("name") not in ("opm_flow", "eclipse"):
        raise ConfigError(f"PET runs decks (OPM Flow, Eclipse); this case uses "
                          f"{simulator.get('name')!r}")  # fmt: skip
    options = dict(simulator.get("options", {}))
    dropped = sorted(set(options) - SIMULATOR_OPTIONS)
    if dropped:
        plan.notes.append(f"the simulator's options {', '.join(dropped)} are APEX's: give PET's "
                          "simulator its own settings")  # fmt: skip
    if options.get("template_values"):
        raise ConfigError("PET's runfile has no template values: write them into the template")
    template = base / str(options.get("deck_template_file", ""))
    if not template.is_file():
        raise ConfigError(f"deck template not found: {template.name}")
    case_dir = Path(str(options["case_dir"])) if options.get("case_dir") else template.parent
    if not case_dir.is_absolute():
        case_dir = base / case_dir
    included = case_dir_files(template)
    if included is None:
        raise ConfigError("the template computes an included file's path, or names one "
                          "outside its case folder: PET's runfile needs fixed paths")  # fmt: skip
    for relative in included:
        origin = case_dir / relative
        if not origin.is_file():
            raise ConfigError(f"{template.name}: included file not found: {relative}")
        plan.copies[relative] = origin
    source = template.read_text()
    if '"${case_dir}/' in source:
        # The importer reads PET's '../x' includes in single quotes only.
        raise ConfigError("write the template's includes in single quotes ('${case_dir}/x')")
    text = source.replace("${case_dir}/", "../")
    if "${case_dir}" in text or _MEMBER.search(text):
        raise ConfigError("the template uses case_dir or member_id beyond its includes, which "
                          "PET's runfile does not have")  # fmt: skip
    plan.texts[f"{RUNFILE}.mako"] = text
    found: dict[str, Any] = {"runfile": RUNFILE}
    workers = dict(execution.get("options", {})).get("workers")
    if workers is not None:
        found["parallel"] = int(workers)
    if "timeout_s" in options:
        found["sim_limit"] = float(options["timeout_s"])
    if simulator.get("name") == "eclipse":
        plan.notes.append("an Eclipse case: give PET's simulator its Eclipse settings")
    return found
