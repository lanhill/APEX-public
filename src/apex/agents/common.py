"""What the runtime agents share: one decision, by the model or by the rules.

:func:`decide` runs a task through :class:`~apex.agents.loop.AgentLoop` with read-only
tools, or, without a model, records the rules' decision. The agents then hold the model's
answer to their facts (each its own ``reconcile``).
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel

from apex.agents.loop import SYSTEM, AgentLoop, Fallback, LoopSettings
from apex.agents.trace import CallRecorder
from apex.algorithms.data_assimilation.registries import methods
from apex.core.errors import PluginError
from apex.core.llm import LLMProvider
from apex.services.api import ApexService
from apex.services.tools import ToolSession

#: The tools the diagnosis, planning and report agents may use: reading only.
READ_ONLY_TOOLS = ("run_summary", "list_runs", "list_plugins", "get_schema", "compare_study")

DecidedBy = Literal["model", "fallback", "rules"]


@dataclass(frozen=True)
class Decision[T: BaseModel]:
    """A decision and who made it.

    Attributes
    ----------
    value
        The validated decision.
    decided_by
        ``model``, ``fallback`` (the model failed; the rules decided) or ``rules`` (no model).
    reason
        Why the fallback decided.
    trace
        The decision's trace file.
    unavailable
        Whether the fallback decided because the provider was unavailable (it could not be
        reached, a usage limit, a timeout), not for another failure of the model.
    """

    value: T
    decided_by: DecidedBy
    reason: str | None
    trace: Path
    unavailable: bool = False


def decide[T: BaseModel](
    service: ApexService,
    provider: LLMProvider | None,
    recorder: CallRecorder,
    *,
    task: str,
    response_model: type[T],
    rules: T,
    rule: str,
    thresholds: Mapping[str, Any],
    client: str,
    settings: LoopSettings | None = None,
    tools: Sequence[str] = READ_ONLY_TOOLS,
    system: str = SYSTEM,
) -> Decision[T]:
    """Return the model's validated decision on ``task``, or the rules' (see the module)."""
    if provider is None:
        trace = recorder.record_decision(
            None,
            task=task,
            decision=rules,
            fell_back=False,
            reason=None,
            rule=rule,
            thresholds=thresholds,
            tool_calls=[],
        )
        return Decision(rules, "rules", None, trace)
    loop = AgentLoop(
        provider,
        recorder,
        session=ToolSession(service, client=client, model=provider.model_id),
        tools=tools,
        settings=settings or LoopSettings(max_turns=6),
    )
    outcome = loop.run(
        task, response_model, Fallback(rule, lambda: rules, thresholds), system=system
    )
    decided_by: DecidedBy = "fallback" if outcome.fell_back else "model"
    return Decision(outcome.decision, decided_by, outcome.reason, outcome.trace,
                    outcome.unavailable)  # fmt: skip


def close(value: float, fact: float, rtol: float) -> bool:
    """Whether ``value`` matches ``fact`` within ``rtol`` (or both are about zero)."""
    return abs(value - fact) <= rtol * max(abs(fact), abs(value), 1e-12)


def default_localization(method: str) -> str:
    """Return the adaptive localization ``method`` takes: ``adaptive`` unless it says otherwise.

    LM-EnRML and GN-EnRML localize only in the approx update's projected-data space
    (``adaptive_projected``, upstream PET's), so proposing ``adaptive`` there would fail.
    """
    try:
        factory = methods.factory(method)
    except PluginError:
        return "adaptive"
    return str(getattr(factory, "default_localization", "adaptive"))
