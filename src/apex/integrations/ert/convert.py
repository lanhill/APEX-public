"""An ERT case as a native APEX case, run without ERT (``apex ert convert``).

Cases whose forward model is one step of ``FLOW``, ``ECLIPSE100`` or ``ECLIPSE300`` on a
``DATA_FILE`` (with ``RUN_TEMPLATE``s) become an APEX configuration on ``opm_flow`` or
``eclipse``:

- **the deck**: the ``DATA_FILE`` becomes ``model/deck.mako``. Each ``INCLUDE`` of a file
  ERT writes into the run path is inlined: a ``FIELD``'s values from the state, and a
  ``RUN_TEMPLATE``'s (or a ``GEN_KW`` template's) converted text. ``INCLUDE``s of the case's
  own static files point at them. ERT's substitutions become Mako: ``<IENS>`` the member id,
  ``<ITER>`` 0, ``DEFINE``/``DATA_KV`` and the built-in keys their values, and each
  ``GEN_KW`` key its value, ERT's transform of the latent state written into the template;
- **the state**: each ``FIELD`` (GRDECL, no transforms) a gridded variable, its prior the
  ``INIT_FILES``; each ``GEN_KW`` group a variable of its keys' latent standard-normal values,
  its prior ERT's own samples (the case's ``RANDOM_SEED``); together, a ``file`` prior;
- **the observations**: summary observations, as APEX's CSV.

What APEX cannot run natively is refused, with a message: other forward-model steps,
``SURFACE``, fields in other formats or with transforms, ``FORWARD_INIT``, other observation
types, ``GEN_KW`` distributions beyond normal, lognormal, uniform, logunif,
truncated_normal, const and raw. Such cases stay on ``APEX_HISTORY_MATCH`` (phase B2), which
runs any forward model.
"""

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import tomli_w

from apex.core.errors import ConfigError

#: The forward-model steps a native case can stand in for, and APEX's simulator for each.
SIMULATOR_STEPS = {"FLOW": "opm_flow", "ECLIPSE100": "eclipse", "ECLIPSE300": "eclipse"}
#: ERT's GEN_KW distributions the converted deck reproduces (``transform_numpy``, the same in
#: ERT 25 and 26); the others, such as ERT 26's ``pert``, are refused.
DISTRIBUTIONS = ("normal", "lognormal", "uniform", "logunif", "truncated_normal", "const", "raw")
#: ``INCLUDE 'file' /`` in an Eclipse deck (quotes optional, over lines).
INCLUDE = re.compile(r"INCLUDE\s+['\"]?([^'\"\s/]+)['\"]?\s*/", re.IGNORECASE)
#: An ERT substitution key.
KEY = re.compile(r"<[A-Za-z_][A-Za-z0-9_]*>")

#: The template's Python helpers: ERT's GEN_KW transforms (``distribution.py``).
HELPERS = """<%!
import math

def _ndtr(x):
    return 0.5 * math.erfc(-x / math.sqrt(2.0))

def gen_kw(x, kind, p):
    if kind == "normal":
        return x * p["std"] + p["mean"]
    if kind == "lognormal":
        return math.exp(x * p["std"] + p["mean"])
    if kind == "uniform":
        return _ndtr(x) * (p["max"] - p["min"]) + p["min"]
    if kind == "logunif":
        lo, hi = math.log(p["min"]), math.log(p["max"])
        return math.exp(lo + _ndtr(x) * (hi - lo))
    if kind == "truncated_normal":
        return min(max(x * p["std"] + p["mean"], p["min"]), p["max"])
    if kind == "const":
        return p["value"]
    return x
%>"""


@dataclass
class Converted:
    """What :func:`convert_case` wrote, and what it noted."""

    config: Path
    template: Path
    notes: list[str] = field(default_factory=list)


def convert_case(
    case_file: Path, out_dir: Path, *, method: str = "esmda", overwrite: bool = False
) -> Converted:
    """Write the native APEX case of an ERT case into ``out_dir`` (see the module docstring).

    Parameters
    ----------
    case_file
        The ERT case (``.ert``).
    out_dir
        Where ``config.toml``, ``model/deck.mako``, ``prior.npz`` and ``observations.csv``
        go.
    method
        APEX's method in the configuration (``esmda`` with ``[4, 4, 4, 4]`` by default; edit
        it as for any case).
    overwrite
        Replace a case already in ``out_dir``.

    Raises
    ------
    ConfigError
        If the case has what a native APEX case cannot run, or ``out_dir`` is not empty.
    """
    from apex.integrations.ert.case import _init_files, _write_observations, read_ert_case
    from apex.integrations.ert.config import parse_case
    from apex.integrations.ert.parameters import gen_kw_keys

    case_file = case_file.resolve()
    if out_dir.exists() and any(out_dir.iterdir()) and not overwrite:
        raise ConfigError(f"{out_dir} is not empty; pass overwrite to replace the case there")
    info = read_ert_case(case_file)
    config = parse_case(case_file)
    problems: list[str] = []
    steps = [s.name for s in config.forward_model_steps]
    if len(steps) != 1 or steps[0] not in SIMULATOR_STEPS:
        problems.append(
            f"the forward model is {steps}; a native case runs one of "
            f"{', '.join(SIMULATOR_STEPS)} (keep other cases on APEX_HISTORY_MATCH)"
        )
    unsupported = [f"{p.name} ({p.kind})" for p in info.parameters if p.kind != "gen_kw"]
    if unsupported:
        problems.append(
            f"parameters {', '.join(unsupported)}: a native case takes GRDECL FIELDs without "
            "transforms and GEN_KW"
        )
    other = sorted({o.kind for o in info.observations if o.kind != "summary"})
    if other:
        problems.append(f"observation types {other}: a native case takes summary observations")
    groups: dict[str, list[tuple[str, str, dict[str, float]]]] = {}
    for parameter in info.parameters:
        if parameter.kind != "gen_kw":
            continue
        keys = []
        for key in gen_kw_keys(config, parameter.name):
            settings = key.distribution
            kind = str(settings.name)
            if kind not in DISTRIBUTIONS:
                problems.append(f"GEN_KW {parameter.name}:{key.name} has distribution {kind}")
                continue
            values = {k: float(v) for k, v in settings.model_dump().items() if k != "name"}
            keys.append((str(key.name), kind, values))
        groups[parameter.name] = keys
    data_file = _data_file(config, info.eclbase)
    if data_file is None:
        problems.append("the case has no DATA_FILE")
    if problems:
        raise ConfigError(f"{case_file} cannot be a native APEX case: " + "; ".join(problems))
    assert data_file is not None
    out_dir.mkdir(parents=True, exist_ok=True)
    model = out_dir / "model"
    model.mkdir(exist_ok=True)
    notes: list[str] = []

    substitutions = {str(k): str(v) for k, v in dict(config.substitutions).items()}
    gen_kw_keys_map = {
        f"<{key}>": (group, i, kind, values)
        for group, keys in groups.items()
        for i, (key, kind, values) in enumerate(keys)
    }
    templates = {Path(dst).name: Path(src) for src, dst in config.ert_templates}
    field_files = {f.output_file: f for f in info.fields}
    inlined: set[str] = set()

    def converted(text: str, where: str, depth: int = 0) -> str:
        def inline(match: re.Match[str]) -> str:
            name = match.group(1)
            if name in field_files:
                variable = field_files[name].name.lower()
                return (
                    f"{field_files[name].name}\n% for v in {variable}:\n"
                    '${"%.10e" % v}\n% endfor\n/'
                )
            if name in templates and templates[name] != Path(data_file) and depth < 5:
                inlined.add(name)
                return converted(templates[name].read_text(), name, depth + 1)
            static = case_file.parent / name
            if static.exists():
                return f"INCLUDE\n  '{static}' /"
            notes.append(f"{where}: INCLUDE {name} is neither a file of the case nor written by it")
            return match.group(0)

        # The deck's own text is made literal first; then the inlined blocks and the
        # substitutions add Mako.
        text = INCLUDE.sub(inline, _escape(text))

        def substitute(match: re.Match[str]) -> str:
            key = match.group(0)
            if key == "<IENS>":
                return "${member_id}"
            if key == "<ITER>":
                return "0"
            if key in gen_kw_keys_map:
                group, i, kind, values = gen_kw_keys_map[key]
                return f"${{'%.10g' % gen_kw({group.lower()}[{i}], {kind!r}, {values!r})}}"
            if key in substitutions:
                return substitutions[key]
            notes.append(f"{where}: {key} has no value; left as it is")
            return key

        return KEY.sub(substitute, text)

    deck = converted(Path(data_file).read_text(), Path(data_file).name)
    for name in sorted(set(templates) - inlined - {Path(data_file).name}):
        notes.append(
            f"{name} (a template ERT writes) is not INCLUDEd by the deck; a native run does "
            "not write it"
        )
    template = model / "deck.mako"
    template.write_text(
        f"-- Converted by apex ert convert from {case_file}.\n" + HELPERS + "\n" + deck
    )

    prior: dict[str, Any] = dict(_init_files(info))
    for parameter in info.parameters:
        if parameter.prior is not None:
            prior[parameter.name.lower()] = parameter.prior
    np.savez(out_dir / "prior.npz", **prior)
    _write_observations(out_dir / "observations.csv", info.observations)
    step = steps[0]
    simulator: dict[str, Any] = {"deck_template_file": "model/deck.mako"}
    if step != "FLOW":
        simulator["version"] = "e300" if step == "ECLIPSE300" else "e100"
        version = substitutions.get("<VERSION>")
        if version:
            simulator["eclrun_args"] = [f"--version={version}"]
        else:
            notes.append("ECLIPSE: no <VERSION> in the case; give eclrun_args a --version")
    variables = [
        {"name": f.name.lower(), "size": int(np.prod(f.grid)), "grid": list(f.grid),
         **({"lower": f.lower} if f.lower is not None else {}),
         **({"upper": f.upper} if f.upper is not None else {})}
        for f in info.fields
    ] + [
        {"name": p.name.lower(), "size": p.size} for p in info.parameters if p.kind == "gen_kw"
    ]  # fmt: skip
    document = {
        "run": {"name": case_file.stem, "seed": int(info.seed or 0)},
        "state": {"variables": variables},
        "prior": {
            "ensemble_size": info.realizations,
            "generator": {"name": "file", "options": {"ensemble_file": "prior.npz"}},
        },
        "observations": {"file": "observations.csv"},
        "case": {"kind": "real"},
        "simulator": {"name": SIMULATOR_STEPS[step], "options": simulator},
        "execution": {"name": "process", "options": {"workers": info.max_running}},
        "method": {"name": method}
        | ({"options": {"inflation": [4, 4, 4, 4]}} if method == "esmda" else {}),
    }
    path = out_dir / "config.toml"
    path.write_text(
        f"# Converted by apex ert convert from {case_file}.\n" + tomli_w.dumps(document)
    )
    (out_dir / "convert.json").write_text(
        json.dumps({"case": str(case_file), "notes": notes}, indent=2) + "\n"
    )
    return Converted(path, template, notes)


def _data_file(config: Any, eclbase: str) -> str | None:  # noqa: ANN401 (ERT's ErtConfig)
    """Return the ``DATA_FILE``: the template ERT writes as ``<ECLBASE>.DATA``."""
    for source, target in config.ert_templates:
        if Path(target).name in (f"{eclbase}.DATA", "<ECLBASE>.DATA") or str(target).endswith(
            ".DATA"
        ):
            return str(source)
    return None


def _escape(text: str) -> str:
    """Make deck text literal for Mako: ``%`` and ``##`` at line starts, and ``${``."""
    lines = []
    for line in text.splitlines():
        stripped = line.lstrip()
        if stripped.startswith("%"):
            line = line.replace("%", "%%", 1)
        elif stripped.startswith("##"):
            line = line.replace("##", "${'#'}#", 1)
        lines.append(line.replace("${", "${'$'}{"))
    return "\n".join(lines) + ("\n" if text.endswith("\n") else "")
