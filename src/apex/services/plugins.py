"""Capability discovery: every registered plugin, with the schema of its options.

Option schemas come from each factory's typed signature (APEX is strict-mypy, so every
option is annotated): the parameters become a strict pydantic model, whose JSON Schema
clients read and against which a configuration's options are checked before a run.
Parameters that APEX supplies itself (e.g. a method's state bounds) are listed apart.

Each option that can hold a string declares what it names (``core.options``);
:func:`option_roles` returns those declarations for the service's ``[policy]``, with the
options the composition reads itself before it builds the plugin (:data:`COMPOSED`).
"""

import inspect
import typing
from collections.abc import Callable, Mapping
from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError, create_model
from pydantic.errors import PydanticSchemaGenerationError, PydanticUserError

import apex.services.composition  # noqa: F401  (registers every built-in plugin)
from apex.algorithms.data_assimilation.registries import (
    damping_schemes,
    gmac_constraint_metrics,
    gmac_constraints,
    gmac_norms,
    gmac_transforms,
    localizations,
    methods,
    update_strategies,
)
from apex.algorithms.geophysics.registries import (
    avo_attributes,
    displacement_kernels,
    petro_elastic_models,
    reflectivity,
)
from apex.algorithms.optimization import objectives, optimizers, response_objectives, step_rules
from apex.core import plugins
from apex.core.audience import audience
from apex.core.errors import PluginError
from apex.core.options import INPUT, TEXT, Role, role_of
from apex.core.registry import Registry
from apex.domain.config import HistoryMatchConfig, PluginSection
from apex.domain.optimization import OptimizationConfig
from apex.domain.services import PluginInfo

#: The registries clients can configure, by the name used in schemas and errors.
REGISTRIES: dict[str, Registry[Any]] = {
    "methods": methods,
    "update_strategies": update_strategies,
    "localizations": localizations,
    "damping_schemes": damping_schemes,
    "gmac_transforms": gmac_transforms,
    "gmac_norms": gmac_norms,
    "gmac_constraints": gmac_constraints,
    "gmac_constraint_metrics": gmac_constraint_metrics,
    "priors": plugins.priors,
    "simulators": plugins.simulators,
    "executors": plugins.executors,
    "data_transforms": plugins.data_transforms,
    "parameterizations": plugins.parameterizations,
    "model_error_models": plugins.model_error_models,
    "metrics": plugins.metrics,
    "stopping_rules": plugins.stopping_rules,
    "forecast_scores": plugins.forecast_scores,
    "optimizers": optimizers,
    "objectives": objectives,
    "response_objectives": response_objectives,
    "step_rules": step_rules,
    "figures": plugins.figures,
    "figure_presets": plugins.figure_presets,
    # the geophysical forward models' parts, configured with a geophysics
    # simulator (phase 2).
    "petro_elastic_models": petro_elastic_models,
    "reflectivity": reflectivity,
    "avo_attributes": avo_attributes,
    # Phase 3: seafloor displacement's kernels.
    "displacement_kernels": displacement_kernels,
}

#: The workflow each registry serves, for front ends that group plugins (the web UI).
WORKFLOWS: dict[str, str] = {
    **dict.fromkeys(
        (
            "methods",
            "update_strategies",
            "localizations",
            "damping_schemes",
            "gmac_transforms",
            "gmac_norms",
            "gmac_constraints",
            "gmac_constraint_metrics",
            "priors",
            "data_transforms",
            "parameterizations",
            "model_error_models",
            "metrics",
            "stopping_rules",
        ),
        "history_matching",
    ),
    "forecast_scores": "prediction",
    **dict.fromkeys(
        ("optimizers", "objectives", "response_objectives", "step_rules"), "optimization"
    ),
    "simulators": "simulation",
    "executors": "simulation",
    **dict.fromkeys(
        ("petro_elastic_models", "reflectivity", "avo_attributes", "displacement_kernels"),
        "simulation",
    ),
    "figures": "reporting",
    "figure_presets": "reporting",
}

#: Plugins that serve another workflow than their registry's: the control maps of
#: optimizations are ``parameterizations``, as history matching's are.
PLUGIN_WORKFLOWS: dict[tuple[str, str], str] = {
    ("parameterizations", "control_schedule"): "optimization",
    ("parameterizations", "control_scaling"): "optimization",
}

#: Parameters the composition supplies (not settable in a configuration's options).
SUPPLIED: dict[str, frozenset[str]] = {
    "methods": frozenset(
        {
            "strategy",
            "lower_bounds",
            "upper_bounds",
            "schedule",
            "max_iterations",
            "inner_method",
            "block_sizes",
        }
    ),
    "update_strategies": frozenset({"localization", "state_variables", "property_map"}),
    "localizations": frozenset({"block_sizes", "variable_names"}),
    "executors": frozenset({"pools"}),
}


#: Options the composition reads itself and does not pass on (``services/composition.py``):
#: a file it loads into the plugin's array, and the key of that array in the file.
COMPOSED: dict[str, dict[str, Role]] = {
    # A wrapping method's (CHOP) inner method, built by the composition.
    "methods": {"inner": TEXT},
    "metrics": {"actnum_file": INPUT, "actnum_key": TEXT},
    "priors": {"mean_file": INPUT, "mean_key": TEXT},
    "data_transforms": {"mask_file": INPUT, "mask_key": TEXT},
}


def _parameters(factory: Callable[..., object]) -> list[inspect.Parameter]:
    target = factory.__init__ if inspect.isclass(factory) else factory
    try:
        # include_extras keeps Annotated's roles; pydantic ignores them.
        hints = typing.get_type_hints(target, include_extras=True)
    except (NameError, TypeError):
        hints = {}
    result = []
    for parameter in inspect.signature(factory).parameters.values():
        if parameter.kind in (parameter.VAR_POSITIONAL, parameter.VAR_KEYWORD):
            continue
        annotation = hints.get(parameter.name, parameter.annotation)
        result.append(parameter.replace(annotation=annotation))
    return result


def option_model(registry: str, name: str) -> type[BaseModel]:
    """Return the strict model of a plugin's configurable options.

    Raises
    ------
    KeyError
        If the registry is unknown.
    PluginError
        If the plugin is unknown.
    """
    factory = REGISTRIES[registry].factory(name)
    supplied = SUPPLIED.get(registry, frozenset())
    fields: dict[str, Any] = {}
    for parameter in _parameters(factory):
        if parameter.name in supplied:
            continue
        annotation = (
            Any if parameter.annotation is inspect.Parameter.empty else parameter.annotation
        )
        default = ... if parameter.default is inspect.Parameter.empty else parameter.default
        fields[parameter.name] = (annotation, default)
    if registry == "methods" and "inner_method" in {p.name for p in _parameters(factory)}:
        # A wrapper (CHOP) takes its wrapped method by name; the composition builds it.
        fields["inner"] = (
            Annotated[str, TEXT],
            Field(default="gies", description="The wrapped method's name (its options follow)."),
        )
    config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)
    model: type[BaseModel] = create_model(f"{registry}.{name}", __config__=config, **fields)
    return model


def _schema(model: type[BaseModel]) -> dict[str, Any]:
    try:
        return model.model_json_schema()
    except (PydanticSchemaGenerationError, PydanticUserError, TypeError):
        # A parameter of a type JSON cannot describe (an array, an object): describe the
        # others, and mark these as given by other configuration or in Python only.
        properties: dict[str, Any] = {}
        for field_name, info in model.model_fields.items():
            try:
                one: dict[str, Any] = {field_name: (info.annotation, None)}
                single = create_model(
                    "one", __config__=ConfigDict(arbitrary_types_allowed=True), **one
                )
                properties[field_name] = single.model_json_schema()["properties"][field_name]
            except (PydanticSchemaGenerationError, PydanticUserError, TypeError):
                properties[field_name] = {"description": f"{info.annotation} (not JSON)"}
        return {"type": "object", "properties": properties, "additionalProperties": False}


def plugin_info(registry: str, name: str) -> PluginInfo:
    """Describe one plugin."""
    entries = REGISTRIES[registry]
    factory = entries.factory(name)
    status = getattr(factory, "status", None)
    doc = inspect.getdoc(factory) or ""
    # A name registered for a factory an earlier name already has is that plugin's alias.
    first = next(o for o in entries.registered() if entries.factory(o) is factory)
    summary = doc.splitlines()[0] if doc else ""
    if first != name:
        summary = f"An alias of {first!r}. {summary}".rstrip()
    return PluginInfo(
        registry=registry,
        name=name,
        workflow=PLUGIN_WORKFLOWS.get((registry, name), WORKFLOWS.get(registry)),  # type: ignore[arg-type]
        status=status if isinstance(status, str) else None,
        summary=summary,
        options=_schema(option_model(registry, name)),
        supplied=sorted(
            p.name for p in _parameters(factory) if p.name in SUPPLIED.get(registry, frozenset())
        ),
        visibility="private" if is_private(registry, name) else "public",
        unavailable=_unavailable(factory),
    )


def _unavailable(factory: object) -> str | None:
    """Why a plugin cannot be used here: a module of its ``requires`` that is not installed.

    ``requires`` maps importable module names to the extra providing them; the modules are
    looked up, not imported.
    """
    import importlib.util

    for module, extra in (getattr(factory, "requires", None) or {}).items():
        try:
            found = importlib.util.find_spec(module) is not None
        except (ImportError, ValueError):
            found = False
        if not found:
            return f"needs {module} (the {extra} extra)"
    return None


def is_private(registry: str, name: str) -> bool:
    """Whether a registered plugin is private (unknown: no).

    Private: its ``visibility = "private"`` attribute, or a distribution whose audience is
    not public (:func:`audience`). Private plugins stay out of the public
    documentation and web-UI guests never see or use them (the user, 2026-10-13).
    """
    entries = REGISTRIES.get(registry)
    if entries is None or name not in entries.names():
        return False
    if getattr(entries.factory(name), "visibility", "public") == "private":
        return True
    origin = entries.origin(name)
    return origin is not None and audience(origin) != "public"


def all_plugins() -> list[PluginInfo]:
    """Describe every registered plugin, by registry and name."""
    return [
        plugin_info(registry, name)
        for registry, entries in REGISTRIES.items()
        for name in entries.names()
    ]


def geophysics_sections(config: HistoryMatchConfig) -> list[tuple[str, PluginSection]]:
    """Return ``[geophysics]``'s plugin sections, each with its registry's name."""
    section = config.geophysics
    if section is None:
        return []
    sections: list[tuple[str, PluginSection]] = []
    if section.pem is not None:
        # An option set by a state variable ({ state = ... }) is checked by the composition;
        # any other table (standard's fluid_fits) is an ordinary option.
        fixed = {
            k: v
            for k, v in section.pem.options.items()
            if not (isinstance(v, dict) and "state" in v)
        }
        sections.append(("petro_elastic_models", section.pem.model_copy(update={"options": fixed})))
    for datum in section.data:
        if datum.kind == "avo":
            options = datum.avo()
            sections.append(("reflectivity", options.reflectivity))
            attribute = "trace" if options.form == "traces" else options.attribute
            sections.append(("avo_attributes", PluginSection(name=attribute)))
        elif datum.kind == "seafloor_displacement":
            sections.append(("displacement_kernels", datum.displacement().kernel))
    return sections


#: A wavelet transform's layout options, which the composition gives an AVO field itself.
_LAYOUT = frozenset({"grid", "mask_file", "mask", "order"})


def _transform_section(config: HistoryMatchConfig, section: PluginSection) -> PluginSection:
    """Return a data transform's section as the plugin receives it.

    A ``wavelet`` entry for an AVO field of ``[geophysics]`` that gives no layout gets the
    traces' own from the composition, so its ``grid`` counts as given;
    the composition checks the layout itself (sparse lattices, partial layouts).
    """
    geophysics = config.geophysics
    if section.name != "wavelet" or geophysics is None or _LAYOUT & set(section.options):
        return section
    fields = {name for datum in geophysics.data for name in datum.datatypes()}
    if section.options.get("datatype") not in fields:
        return section
    return section.model_copy(update={"options": {**section.options, "grid": [1]}})


def plugin_sections(config: HistoryMatchConfig) -> list[tuple[str, PluginSection]]:
    """Return a configuration's plugin sections, each with its registry's name."""
    sections: list[tuple[str, PluginSection]] = [
        ("methods", config.method),
        ("simulators", config.simulator),
        ("executors", config.execution),
        ("priors", config.prior.generator),
        *(("priors", section) for section in config.prior.variables.values()),
        *(("priors", section) for section in config.prior.groups.values()),
        *(("data_transforms", _transform_section(config, s)) for s in config.data_transforms),
        *(("parameterizations", section) for section in config.parameterizations),
        *((("model_error_models", config.model_error),) if config.model_error else ()),
        *geophysics_sections(config),
        # "data_mismatch" names the objective, always recorded, not a plugin.
        *(("metrics", s) for s in config.metrics or [] if s.name != "data_mismatch"),
    ]
    if config.method.strategy is not None:
        sections.append(("update_strategies", config.method.strategy))
    if config.method.localization is not None:
        sections.append(("localizations", config.method.localization))
    if config.method.damping is not None:
        sections.append(("damping_schemes", config.method.damping))
    if config.validation is not None:
        sections.append(("stopping_rules", config.validation.stopping_rule()))
    return sections


def option_roles(registry: str, name: str) -> dict[str, Role | None]:
    """Return what each option of a plugin names (``None``: undeclared), composed included.

    Raises
    ------
    KeyError
        If the registry is unknown.
    PluginError
        If the plugin is unknown.
    """
    model = option_model(registry, name)
    roles: dict[str, Role | None] = {
        field: role_of(info.rebuild_annotation()) for field, info in model.model_fields.items()
    }
    return {**roles, **COMPOSED.get(registry, {})}


def wrapped(
    registry: str, name: str, options: Mapping[str, Any]
) -> tuple[dict[str, Any], str, dict[str, Any]] | None:
    """Split a wrapping method's options (CHOP): its own, the inner's name, the rest.

    ``None`` when the plugin wraps nothing. As the composition does
    (``composition._wrapped_method``): the wrapper's own options are those its signature
    names, ``inner`` names the wrapped method (default ``gies``), the rest are the inner's.
    """
    if registry != "methods" or name not in REGISTRIES[registry].names():
        return None
    parameters = inspect.signature(REGISTRIES[registry].factory(name)).parameters
    if "inner_method" not in parameters:
        return None
    own = {k: v for k, v in options.items() if k in parameters}
    rest = {k: v for k, v in options.items() if k not in parameters and k != "inner"}
    return own, str(options.get("inner", "gies")), rest


def iterated_method(name: str, options: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
    """Return the method that runs a method's steps, and its options.

    A wrapper's (CHOP's) inner method, followed through every wrapper; the method itself
    otherwise. Whether ``max_iterations`` applies, or ES-MDA's inflation sets the steps, is
    this method's, not the wrapper's.
    """
    split = wrapped("methods", name, options)
    if split is None:
        return name, dict(options)
    _, inner, rest = split
    return iterated_method(inner, rest)


def control_name(name: str, options: Mapping[str, Any]) -> str:
    """Return what an iterative method's per-trial parameter is called (``damping`` default).

    ``IterationRecord.damping`` holds GIES's and LM-EnRML's damping lambda but GN-EnRML's step
    length gamma; tables and figures label it by the method that runs the steps
    (``control_name``), so a step length is not shown as a damping.
    """
    target, _ = iterated_method(name, options)
    if target not in REGISTRIES["methods"].names():
        return "damping"
    label = getattr(REGISTRIES["methods"].factory(target), "control_name", "damping")
    return label if isinstance(label, str) else "damping"


def takes_max_iterations(name: str, options: Mapping[str, Any]) -> bool:
    """Return whether a method (or the method a wrapper runs) takes ``max_iterations``."""
    target, _ = iterated_method(name, options)
    if target not in REGISTRIES["methods"].names():
        return False
    return "max_iterations" in inspect.signature(REGISTRIES["methods"].factory(target)).parameters


def check_options(registry: str, name: str, options: Mapping[str, Any]) -> list[str]:
    """Return what is wrong with a plugin's options (empty when they are valid).

    The options the composition reads itself (:data:`COMPOSED`) are accepted: they used to
    be refused here, so the service refused configurations ``apex run`` accepted. A
    wrapping method's options are checked as its own and its inner method's.
    """
    split = wrapped(registry, name, options)
    if split is not None:
        own, inner, rest = split
        return _check_model(registry, name, own) + check_options(registry, inner, rest)
    return _check_model(registry, name, options)


def _check_model(registry: str, name: str, options: Mapping[str, Any]) -> list[str]:
    try:
        model = option_model(registry, name)
    except PluginError:
        known = ", ".join(REGISTRIES[registry].names())
        return [f"unknown {registry} plugin {name!r}; available: {known}"]
    composed = COMPOSED.get(registry, {})
    options = {k: v for k, v in options.items() if k not in composed}
    try:
        model.model_validate(dict(options))
    except ValidationError as error:
        return [
            f"{registry} {name!r}: {'.'.join(str(p) for p in e['loc']) or 'options'}: {e['msg']}"
            for e in error.errors()
        ]
    return []


def objective_plugins(config: OptimizationConfig) -> list[tuple[str, PluginSection]]:
    """Return an optimization's objectives with their registries.

    ``[objective]`` or each entry of ``[[objectives]]``: ``response_objectives`` for a
    simulated one, ``objectives`` otherwise. Used wherever plugins are checked or listed.
    """
    simulated = set(REGISTRIES["response_objectives"].names())
    return [
        ("response_objectives" if section.name in simulated else "objectives", section)
        for section in config.objective_sections
    ]
