import { useState } from "react";

import type { DeletionPlan, DeletionTarget } from "../api";
import { ErrorBox, useApi, useCan } from "./common";

/** Megabytes, for a person. */
export function sizeText(bytes: number): string {
  if (bytes < 1e6) return `${Math.max(1, Math.round(bytes / 1e3))} kB`;
  return `${(bytes / 1e6).toFixed(bytes < 1e8 ? 1 : 0)} MB`;
}

/** The lines a deletion's confirmation lists: what goes, what stays, and what blocks it. */
export function deletionLines(plan: DeletionPlan, target?: DeletionTarget): string[] {
  // Asked about one optimization (or run) that is a study's cell: the whole study goes.
  const cellOf =
    target && "job" in target && plan.kind === "study" && target.job !== plan.job && plan.studies.length > 0;
  const count = plan.optimizations?.length ?? 0;
  const lines = [
    ...(cellOf
      ? [
          `This is a cell of study ${plan.studies.join(", ")}: the whole study goes${
            count ? `, with its ${count} optimization${count === 1 ? "" : "s"}` : ""
          }.`,
        ]
      : []),
    `${plan.jobs.length} job${plan.jobs.length === 1 ? "" : "s"} (${plan.kind} ${plan.job})`,
    ...plan.dirs.map((d) => `folder ${d}`),
    ...plan.studies.map((s) => `study ${s} (its record and comparison)`),
  ];
  if (plan.dirs.length > 0) lines.push(`${sizeText(plan.size)} on disk`);
  return lines;
}

/**
 * Delete one's own work: first the plan of what would go, then a confirmation.
 * Only the server decides what is one's own; this button only asks it.
 */
export function DeleteButton({ target, after }: { target: DeletionTarget; after: string }) {
  const api = useApi();
  const canDelete = useCan("delete");
  const [plan, setPlan] = useState<DeletionPlan>();
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<Error>();
  if (!canDelete) return null;

  const look = async () => {
    setBusy(true);
    setError(undefined);
    try {
      setPlan(await api.deletion(target));
    } catch (err) {
      setError(err instanceof Error ? err : new Error(String(err)));
    } finally {
      setBusy(false);
    }
  };
  const confirm = async () => {
    if (!plan) return;
    setBusy(true);
    setError(undefined);
    try {
      await api.deleteConfirmed(plan.job, plan.token);
      window.location.hash = after;
    } catch (err) {
      setError(err instanceof Error ? err : new Error(String(err)));
      setPlan(undefined);
    } finally {
      setBusy(false);
    }
  };

  if (!plan) {
    return (
      <>
        <button disabled={busy} onClick={() => void look()}>
          Delete…
        </button>
        <ErrorBox error={error} />
      </>
    );
  }
  return (
    <div className="card confirm" role="alertdialog" aria-label="Confirm deletion">
      {plan.blocked.length > 0 ? (
        <>
          <p>
            <strong>This cannot be deleted now:</strong>
          </p>
          <ul>
            {plan.blocked.map((b) => (
              <li key={b}>{b}</li>
            ))}
          </ul>
          <button onClick={() => setPlan(undefined)}>Close</button>
        </>
      ) : (
        <>
          <p>
            <strong>Delete permanently?</strong> This cannot be undone. It deletes:
          </p>
          <ul>
            {deletionLines(plan, target).map((line) => (
              <li key={line}>{line}</li>
            ))}
          </ul>
          {plan.kept.length > 0 && (
            <p className="note">
              Kept: {plan.kept.length} run{plan.kept.length === 1 ? "" : "s"} that another submission also uses.
            </p>
          )}
          <button className="danger" disabled={busy} onClick={() => void confirm()}>
            Delete permanently
          </button>{" "}
          <button disabled={busy} onClick={() => setPlan(undefined)}>
            Cancel
          </button>
        </>
      )}
      <ErrorBox error={error} />
    </div>
  );
}
