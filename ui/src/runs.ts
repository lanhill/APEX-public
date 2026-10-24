// The Runs pages' text: pure functions, so they are unit-tested. Display only.

import { ApiError, type JobHandle, type JobStatus } from "./api";

/** A run's annotations as `key=value` items, each shown apart (one per line), in the server's order. */
export function annotationItems(annotations: Record<string, string>): string[] {
  return Object.entries(annotations).map(([k, v]) => `${k}=${v}`);
}

/** Whether a request failed because what it asked for is not there (or hidden: 404). */
export function isNotFound(error: Error | undefined): boolean {
  return error instanceof ApiError && error.status === 404;
}

/**
 * What a run page says when the run has no report figures yet; `null` (nothing) until the
 * agents and the signed-in user are known, so the text does not change under the reader.
 * With the plotter: its Draw figures button. Without it: the owner gets the command that
 * writes them; anyone else (a web-UI guest, or a user the service could not name, whoami
 * having failed) is told to ask the owner (guests have no command line).
 */
export function noFigures(
  agentsLoaded: boolean,
  canDraw: boolean,
  role: "owner" | "ui" | "pending",
  whoFailed = false,
): { text: string; command: string | null } | null {
  if (!agentsLoaded || (role === "pending" && !whoFailed)) return null;
  if (canDraw) return { text: "No report figures yet: the Draw figures button under Agents above draws them.", command: null };
  if (role === "owner") return { text: "No report figures yet; this command writes them:", command: "apex report RUN_DIR --plots" };
  return { text: "No report figures yet: ask the service's owner to draw them.", command: null };
}

/**
 * What a job page says about an identical run its job reused instead of running; `null` if
 * it reused none. `link`: the run's page, when this user may open it. Else `path`: where
 * the run is (shown workspace-relative), outside the web-UI folder; a guest gets no path.
 */
export function reusedRun(
  status: Pick<JobStatus, "reused" | "reused_visible" | "run_ids">,
): { text: string; link: string | null; path: string | null } | null {
  if (status.reused_visible === undefined || status.reused_visible === null) return null;
  const run = status.run_ids?.[0] ?? null;
  if (status.reused_visible && run) return { text: "Reused an identical run:", link: run, path: null };
  if (status.reused) {
    return {
      text: "Reused an identical run outside your web-UI folder (not listed under Runs), in",
      link: null,
      path: status.reused,
    };
  }
  return { text: "Reused an identical run made outside your web-UI folder.", link: null, path: null };
}

/**
 * What the web UI says after a submission: its job's page to open, or, when the answer is
 * another submission's job that this user may not see (an identical run already queued or
 * running elsewhere), that it is, with no link (its page would be "not found") and no path.
 */
export function submittedNote(handle: Pick<JobHandle, "id" | "reused" | "job_visible" | "job_state">): {
  link: string | null;
  text: string;
} {
  if (handle.job_visible === false) {
    const running = handle.job_state === "queued" || handle.job_state === "running";
    return {
      link: null,
      text: running
        ? "An identical run is already queued or running outside your web-UI folder: nothing new was submitted."
        : "An identical run exists outside your web-UI folder: nothing new was submitted.",
    };
  }
  return { link: handle.id, text: handle.reused ? " (an identical run existed: reused)" : "" };
}
