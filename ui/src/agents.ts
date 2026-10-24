// Which of APEX's runtime agents the web UI shows, and where each one is used. Display only:
// the agents themselves come from GET /v1/agents.

import type { AgentInfo } from "./api";

export type Visibility = "public" | "private";

/** An agent's visibility: the service's (an older service that sends none: public). */
export function visibilityOf(agent: AgentInfo): Visibility {
  return agent.visibility ?? "public";
}

/**
 * The agents the web UI shows: the service's list, less private agents. Every place that lists
 * or offers agents goes through here, so they all agree.
 */
export function visibleAgents(agents: AgentInfo[]): AgentInfo[] {
  return agents.filter((a) => visibilityOf(a) === "public");
}

export type UiPage = "runs" | "optimizations" | "studies" | "editor";

/** Where in the web UI an agent is used, for a person; agents without an entry have no place yet. */
const WHERE_IN_UI: Record<string, { text: string; pages: UiPage[] }> = {
  monitor: { text: "On a run's or an optimization's page: Check health.", pages: ["runs", "optimizations"] },
  diagnosis: { text: "On a run's or an optimization's page: Diagnose.", pages: ["runs", "optimizations"] },
  report: {
    text: "On a run's or an optimization's page: Write report; on a finished study's page: Write the study's report.",
    pages: ["runs", "optimizations", "studies"],
  },
  plotter: {
    text: "On a run's page: Draw figures (a request in words, or empty for the classic set); they appear under Figures.",
    pages: ["runs"],
  },
  planner: {
    text: "In the Editor, on a saved history-matching or optimization configuration: Plan a study of this case.",
    pages: ["editor"],
  },
  "case-converter": {
    text: "On this page, below: Bring a case (import a deck with its history, or create a synthetic case).",
    pages: [],
  },
  geomodeler: {
    text: "In the Editor, on a saved history-matching configuration: Give it a prior.",
    pages: ["editor"],
  },
};

/** An agent's place in the web UI, if any (own entries only: names come from the service). */
export function whereInUi(name: string): { text: string; pages: UiPage[] } | undefined {
  return Object.prototype.hasOwnProperty.call(WHERE_IN_UI, name) ? WHERE_IN_UI[name] : undefined;
}

/** An agent's name as a heading: `diagnosis` → "Diagnosis", `run-monitor` → "Run monitor". */
export function agentLabel(name: string): string {
  const words = name.replace(/[_-]/g, " ");
  return words.charAt(0).toUpperCase() + words.slice(1);
}
