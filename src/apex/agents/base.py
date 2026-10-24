"""What every APEX runtime agent shares: its recorder and the installed advice.

An agent makes one :class:`AgentContext` with its name, and takes its system prompts and
its ``[agents.<name>]`` thresholds from it:

- its decisions are recorded by the :class:`~apex.agents.trace.CallRecorder` (records in
  ``<root>/agents`` unless the agent keeps them with a run, study or chat);
- installed advisors (:mod:`apex.core.advice`) may change its prompts
  (:meth:`AgentContext.system`) and thresholds (:meth:`AgentContext.thresholds`); the ids of
  the advice applied are recorded with each decision;
- :meth:`AgentContext.skill_advice` gives the advisors' advice for one of its skills, and
  :meth:`AgentContext.report` tells them what its code saw in its work.
"""

from pathlib import Path
from typing import Any

from pydantic import BaseModel

from apex.agents.trace import CallRecorder
from apex.core.advice import (
    AgentEvent,
    advise_prompt,
    advise_skill,
    advise_thresholds,
    report_event,
)


class AgentContext:
    """An agent's recorder, with the installed advice applied.

    Parameters
    ----------
    root
        The services root.
    name
        The agent's name, as in the catalogue and its records.
    directory
        Where its records go; default ``root`` (``<root>/agents``).
    """

    def __init__(self, root: Path, name: str, directory: Path | None = None) -> None:
        self.root = Path(root)
        self.name = name
        self.recorder = CallRecorder(Path(directory or root), name)

    def system(self, prompt: str) -> str:
        """Return ``prompt`` with the advice for the agent, recording its ids."""
        text, ids = advise_prompt(prompt, self.name, self.root)
        self._record(ids)
        return text

    def thresholds[S: BaseModel](self, section: S) -> S:
        """Return the agent's ``[agents.<name>]`` section with the advice on it."""
        updated, ids = advise_thresholds(
            f"agents.{self.name.replace('-', '_')}", section, self.root
        )
        self._record(ids)
        return updated

    def _record(self, ids: list[str]) -> None:
        self.recorder.advice = sorted(set(self.recorder.advice) | set(ids))

    def skill_advice(self, skill: str) -> list[str]:
        """Return the installed advisors' advice for one of the agent's skills.

        The advice's ids are recorded with the decision, as the prompt's are.
        """
        texts, ids = advise_skill(self.name, skill, self.root)
        self._record(ids)
        return texts

    def report(
        self,
        key: str,
        summary: str,
        *,
        skill: str = "",
        occurrence: str = "",
        evidence: dict[str, Any] | None = None,
    ) -> None:
        """Report what the agent's code saw in its work to the installed advisors."""
        event = AgentEvent(key, summary, skill, occurrence, evidence or {})
        report_event(self.name, event, self.root)
