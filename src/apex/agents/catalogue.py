"""APEX's runtime agents, as the chat and the web UI list them.

Whatever reads the catalogue (an installed package's tools too) sees an agent added here
without an edit of its own.

Agents workspace members contribute by entry point (``apex.agents``, ``core.agents``) are
listed after core's.
"""

import logging
from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import Literal

from apex.core.agents import AgentContribution, contributed_agents
from apex.core.agents import AgentInfo as AgentInfo  # re-exported: agents import it from here
from apex.core.audience import audience

logger = logging.getLogger(__name__)

#: The agents of core APEX.
CORE_AGENTS = (
    AgentInfo("monitor", "Checks a run's health: collapse, failures, damping, stagnation, "
              "overfitting, cross validation; recommends continue, stop or investigate.",
              "apex agent monitor RUN", "monitor_run", "0018"),
    AgentInfo("diagnosis", "Explains what went wrong in a run (QA/QC of the posterior, "
              "failed members) and what to change, with validated configuration patches.",
              "apex agent diagnose RUN", "diagnose_run", "0025"),
    AgentInfo("planner", "Plans a study (variants, replicates, metrics) for a case and a "
              "goal within a budget, as a draft you review, revise and launch.",
              "apex agent plan CONFIG --goal TEXT", "plan_study", "0025"),
    AgentInfo("report", "Writes a narrative report of a run or a study, every number "
              "checked against the facts.", "apex agent report RUN | --study NAME",
              "write_report", "0025"),
    AgentInfo("plotter", "Draws a run's result figures: metric box and violin plots, forecast "
              "bands per well, layer maps and histograms, or a whole preset such as "
              "brugge_classic.", "apex agent plot RUN [--request TEXT] [--preset NAME]",
              "plot_run", "0059"),
    AgentInfo("case-converter", "Turns a simulator's case (an OPM Flow or Eclipse deck with "
              "its history) into an APEX history-matching case that validates, with its "
              "observations, data errors, state and prior.",
              "apex agent convert import DECK", None, "0067"),
    AgentInfo("geomodeler", "Gives a case's property fields a prior: continuous Gaussian "
              "fields or facies by proportion, from your values or a reference field, each "
              "checked on a drawn sample.", "apex agent geomodel CONFIG", None, "0068"),
)  # fmt: skip


def merged(
    core: Sequence[AgentInfo], contributions: Sequence[AgentContribution]
) -> tuple[tuple[AgentInfo, ...], frozenset[str]]:
    """Return core's agents and the contributed ones, and the names of those kept private.

    A contributed agent of a package whose audience is not public is private
    whatever it says, and stays so under ``[agents.visibility]``; one that would take a core
    agent's name is skipped.
    """
    names = {a.name for a in core}
    sections = {a.name.replace("-", "_") for a in core}
    found, locked = list(core), set()
    for contribution in contributions:
        info = contribution.info
        # Neither a core agent's name nor its section ("an_agent" is an-agent's).
        if info.name in names or contribution.section in sections:
            logger.warning("the contributed agent %r would replace core's; skipped", info.name)
            continue
        names.add(info.name)
        sections.add(contribution.section)
        dist = contribution.distribution
        if dist is not None and audience(dist) != "public":
            info = replace(info, visibility="private")
            locked.add(info.name)
        found.append(info)
    return tuple(found), frozenset(locked)


#: Every agent: core's, then the contributed ones (``core.agents``).
AGENTS, _AUDIENCE_PRIVATE = merged(CORE_AGENTS, contributed_agents())


def agent_catalogue(
    overrides: Mapping[str, Literal["public", "private"]] | None = None,
) -> tuple[AgentInfo, ...]:
    """Return every agent with its visibility, ``[agents.visibility]`` applied."""
    overrides = overrides or {}
    return tuple(
        a if a.name in _AUDIENCE_PRIVATE
        else replace(a, visibility=overrides.get(a.name, a.visibility))
        for a in AGENTS
    )  # fmt: skip


def public_agents(
    overrides: Mapping[str, Literal["public", "private"]] | None = None,
) -> tuple[AgentInfo, ...]:
    """Return the agents the web UI and the assistant may name."""
    return tuple(a for a in agent_catalogue(overrides) if a.visibility == "public")
