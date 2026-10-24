"""Which update strategies each method accepts, as the composition decides it.

The documentation's method-strategy table is derived from here, so it is what
``apex validate`` and ``apex run`` accept: each pair is built by
:func:`apex.services.composition.build_method` on a one-variable state, without data or a
simulator, and a refused pair keeps the refusal's own words (the release tester,
2026-10-19, met the refusals one configuration at a time).
"""

import inspect
import logging
from collections.abc import Mapping
from typing import Any

import numpy as np

from apex.core.errors import ConfigError, PluginError
from apex.domain.config import MethodSection, PluginSection
from apex.domain.state import StateLayout, StateVariable
from apex.services import composition
from apex.services.plugins import REGISTRIES, is_private

logger = logging.getLogger(__name__)

#: The options a method needs to be built at all (none of them decides a strategy).
_REQUIRED_OPTIONS: dict[str, dict[str, Any]] = {
    "esmda": {"inflation": [1.0]},
    "multilevel_esmda": {"inflation": [1.0], "levels": [{"members": 2, "weight": 1.0}]},
}


#: The start of a pair's answer when it could not be built at all (neither accepted nor
#: refused by a method's own check).
UNKNOWN = "unknown: "


def strategy_compatibility(*, public_only: bool = True) -> dict[str, dict[str, str | None]]:
    """Return ``{strategy: {method: None if accepted, else the refusal}}``.

    A strategy that takes a localization is built with ``adaptive`` when it needs one (the
    local analyses). A strategy with options of its own that are required (``ert_localized``'s
    ``taper_file``) is given to the method unconstructed: the methods decide by the
    strategy's kind, never by its options. With ``public_only`` (the default: the tables are
    for public pages) private and experimental plugins are left out. A pair that fails in
    another way is logged and answered :data:`UNKNOWN` plus the error, so that one plugin's
    fault does not stop the table.
    """
    layout = StateLayout([StateVariable("x", 4)])
    strategies = _listed("update_strategies", public_only)
    methods = _listed("methods", public_only)
    table: dict[str, dict[str, str | None]] = {}
    for strategy in strategies:
        table[strategy] = {method: _answer(method, strategy, layout) for method in methods}
    return table


def default_strategy(
    method: str, options: Mapping[str, Any] | None = None
) -> dict[str, Any] | None:
    """Return the update strategy ``method`` runs when none is given, as a strategy section.

    ``{"name": ..., "options": {...}}``: the registered name, and the options in which the
    method's own default differs from the strategy's (LM-EnRML's and GN-EnRML's approx keeps
    95 % of the energy, approx's own default 98 %). It is returned only when building the
    method with it gives the same strategy as building it with none, so naming it changes
    nothing. ``None`` when that cannot be told (the method needs more to be built, takes no
    strategy, its default is not a registered strategy, or it differs once named).
    """
    update_strategies = REGISTRIES["update_strategies"]
    layout = StateLayout([StateVariable("x", 4)])
    base = {**_REQUIRED_OPTIONS.get(method, {}), **(options or {})}
    try:
        implicit = getattr(
            composition.build_method(MethodSection(name=method, options=base), layout),
            "strategy", None,
        )  # fmt: skip
        if implicit is None:
            return None
        name = next((n for n in update_strategies.registered()
                     if type(implicit) is update_strategies.factory(n)), None)  # fmt: skip
        if name is None:
            return None
        own = update_strategies.create(name)
        differs = {
            key: value
            for key in inspect.signature(type(implicit)).parameters
            if _plain(value := getattr(implicit, key, None)) and value != getattr(own, key, None)
        }
        section = {"name": name, "options": differs}
        named = getattr(
            composition.build_method(
                MethodSection.model_validate(
                    {"name": method, "options": base, "strategy": section}
                ),
                layout,
            ),
            "strategy",
            None,
        )
    except Exception:  # noqa: BLE001 (not told: the caller keeps the entry as given)
        return None
    return section if _settings(named) == _settings(implicit) else None


def _plain(value: object) -> bool:
    """Whether ``value`` is an option a TOML file holds (a number, text or a flag)."""
    return isinstance(value, bool | int | float | str)


def _settings(strategy: object) -> dict[str, object]:
    """Return a strategy's settings in a comparable form.

    Plain values as they are; sequences and arrays by their values; any other object by
    its type, so that a difference in an attribute this cannot see into fails closed only
    where it shows (one build has it, the other does not).
    """

    def comparable(value: object) -> object:
        if value is None or _plain(value):
            return value
        if isinstance(value, np.ndarray):
            return ("array", value.shape, tuple(np.asarray(value).ravel().tolist()))
        if isinstance(value, list | tuple):
            return tuple(comparable(v) for v in value)
        if isinstance(value, np.generic):
            return value.item()
        return ("object", type(value).__name__)

    return {k: comparable(v) for k, v in vars(strategy).items()}


def accepts_localization(method: Mapping[str, Any], localization: str) -> bool:
    """Return whether a method entry (name, options, strategy) takes ``localization``.

    Built as the composition builds it, on a one-variable state; ``False`` also when it
    cannot be told.
    """
    entry = {**method, "localization": {"name": localization}}
    entry["options"] = {**_REQUIRED_OPTIONS.get(str(method.get("name")), {}),
                        **(method.get("options") or {})}  # fmt: skip
    try:
        composition.build_method(
            MethodSection.model_validate(entry), StateLayout([StateVariable("x", 4)])
        )
    except Exception:  # noqa: BLE001 (not told: not proposed)
        return False
    return True


def _listed(registry: str, public_only: bool) -> list[str]:
    names = REGISTRIES[registry].names()
    if not public_only:
        return list(names)
    return [
        name
        for name in names
        if not is_private(registry, name)
        and getattr(REGISTRIES[registry].factory(name), "status", None) != "experimental"
    ]


def _answer(method: str, strategy: str, layout: StateLayout) -> str | None:
    try:
        return _refusal(method, strategy, layout)
    except Exception as error:  # noqa: BLE001 (any plugin's fault: logged, the table goes on)
        logger.warning(
            "could not check method %s with strategy %s: %s",
            method,
            strategy,
            error,
            extra={"event": "compatibility_unknown", "method": method, "strategy": strategy},
        )
        return f"{UNKNOWN}{type(error).__name__}: {error}"


def _refusal(method: str, strategy: str, layout: StateLayout) -> str | None:
    factory = REGISTRIES["update_strategies"].factory(strategy)
    localization = PluginSection(name="adaptive") if _needs_localization(factory) else None
    section = MethodSection(
        name=method,
        options=_REQUIRED_OPTIONS.get(method, {}),
        strategy=PluginSection(name=strategy),
        localization=localization,
    )
    try:
        composition.build_method(section, layout)
    except TypeError:
        if not _required_options(factory):
            raise
        return _unbuilt(method, factory, layout)
    except (ConfigError, ValueError, PluginError) as error:
        return str(error)
    return None


def _unbuilt(method: str, factory: Any, layout: StateLayout) -> str | None:  # noqa: ANN401
    """Ask the method about a strategy whose own options are required, by its kind alone.

    A wrapper (CHOP) is asked with its default inner method (``gies``) built on it, as the
    composition composes it.
    """
    instance = factory.__new__(factory)
    methods = REGISTRIES["methods"]
    try:
        if "inner_method" in _parameters(methods.factory(method)):
            inner = methods.create("gies", strategy=instance)
            methods.create(method, inner_method=inner, block_sizes=[layout.n_state])
        else:
            methods.create(method, strategy=instance, **_REQUIRED_OPTIONS.get(method, {}))
    except (ConfigError, ValueError, PluginError) as error:
        return str(error)
    return None


def _parameters(factory: Any) -> Mapping[str, inspect.Parameter]:  # noqa: ANN401
    return inspect.signature(factory).parameters


def _needs_localization(factory: Any) -> bool:  # noqa: ANN401
    """Whether a strategy takes a localization and cannot run without one."""
    if "localization" not in _parameters(factory):
        return False
    try:
        factory(block_sizes=[4]) if "block_sizes" in _parameters(factory) else factory()
    except ValueError:
        return True
    except TypeError:
        return False
    return False


def _required_options(factory: Any) -> list[str]:  # noqa: ANN401
    """Return a strategy's required options that the composition does not supply."""
    supplied = {"localization", "block_sizes", "state_variables", "property_map"}
    return [
        name
        for name, parameter in _parameters(factory).items()
        if parameter.default is inspect.Parameter.empty
        and name not in supplied
        and parameter.kind not in (parameter.VAR_POSITIONAL, parameter.VAR_KEYWORD)
    ]
