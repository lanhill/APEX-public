// The assistant's proposals: what a card offers to launch, and
// the assumptions an optimization draft was written under. Display only.

import type { Proposal } from "./api";

/** What a proposal launches: a run (a revised run's draft), an optimization or a study. */
export function launchKind(proposal: Pick<Proposal, "kind">): "run" | "optimization" | "study" {
  if (proposal.kind === "launch_run") return "run";
  if (proposal.kind === "launch_optimization") return "optimization";
  return "study";
}

/** Whether a proposal launches an optimization, not a study. */
export function launchesOptimization(proposal: Pick<Proposal, "kind">): boolean {
  return launchKind(proposal) === "optimization";
}

/**
 * The assumptions an optimization draft states in its header, one `# Assumed: ...` line each
 * (`services/optimization_draft.py`): the values the user did not give and should check.
 */
export function assumptionsOf(text: string): string[] {
  const found: string[] = [];
  for (const line of text.split("\n")) {
    const match = /^#\s*Assumed:\s*(.+?)\.?\s*$/.exec(line);
    if (match?.[1]) found.push(match[1]);
    // The header ends at the first line that is not a comment.
    else if (line.trim() && !line.trimStart().startsWith("#")) break;
  }
  return found;
}

/**
 * How a proposed study's arms are not like for like, as the service computed it (the
 * proposal's `data.unmatched`): budgets, a setting both arms take set differently, the case's
 * own settings changed. Shown on the card whatever the answer said (reported issues).
 */
export function unmatchedOf(proposal: Pick<Proposal, "data">): string[] {
  const listed = proposal.data?.unmatched;
  return Array.isArray(listed) ? listed.filter((item): item is string => typeof item === "string") : [];
}
