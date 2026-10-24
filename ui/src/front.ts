// A multi-objective run's front and the catalogue of examples: pure functions, so
// they are unit-tested. Display only: the service computes everything.

import type { Example, OptimizationStep, ParetoFront } from "./api";
import type { Point } from "./chart";
import { OTHER, WORKFLOWS } from "./plugins";

/** One member of a front: its objectives, whether it is the representative, its controls. */
export interface FrontMember {
  index: number;
  objectives: number[];
  representative: boolean;
  feasible: boolean;
  x: number[];
}

/** The front's members, sorted by the first objective (a two-objective front reads left to right). */
export function frontMembers(front: ParetoFront): FrontMember[] {
  return front.objectives
    .map((objectives, index) => ({
      index,
      objectives,
      representative: index === front.representative,
      feasible: (front.violation[index] ?? 0) <= 0,
      x: front.x[index] ?? [],
    }))
    .sort((a, b) => (a.objectives[0] ?? 0) - (b.objectives[0] ?? 0));
}

/**
 * Parallel coordinates: each member's objectives scaled to the front's range per objective
 * (0: the front's best, 1: its worst); an objective with no spread is 0.5.
 */
export function parallelRows(front: ParetoFront): number[][] {
  const n = front.objective_names.length;
  const lo = Array.from({ length: n }, (_, j) => Math.min(...front.objectives.map((f) => f[j] ?? 0)));
  const hi = Array.from({ length: n }, (_, j) => Math.max(...front.objectives.map((f) => f[j] ?? 0)));
  return front.objectives.map((f) =>
    f.map((v, j) => {
      const span = (hi[j] ?? 0) - (lo[j] ?? 0);
      return span > 0 ? (v - (lo[j] ?? 0)) / span : 0.5;
    }),
  );
}

/**
 * Whether a run has several objectives. The service says so from the start
 * (`multi_objective`, `objective_names`); an older one is told by its front, or by its
 * history's hypervolume or `objective:<name>` columns. Such a run has no single objective
 * value: its representative's first objective jumps as the representative changes.
 */
export function isMultiObjective(summary: {
  pareto?: ParetoFront | null;
  history: OptimizationStep[];
  multi_objective?: boolean;
  objective_names?: string[] | null;
}): boolean {
  return (
    summary.multi_objective === true ||
    (summary.objective_names?.length ?? 0) > 1 ||
    Boolean(summary.pareto) ||
    summary.history.some(
      (row) => typeof row.hypervolume === "number" || Object.keys(row).some((k) => k.startsWith("objective:")),
    )
  );
}

/**
 * The objectives whose obj_scaling is negative: quantities to maximise, minimised as their
 * scaled negatives (lower is better). Names from the summary, else from the front. Per
 * [[objectives]] entry, or a single vector [objective]'s one scaling for all of them.
 */
export function maximisedObjectives(summary: {
  objective_names?: string[] | null;
  objective_scalings?: (number | null)[] | null;
  objective_scaling?: number | null;
  pareto?: ParetoFront | null;
}): string[] {
  const names = summary.objective_names ?? summary.pareto?.objective_names ?? [];
  // One vector [objective] (e.g. zdt1): its one obj_scaling scales every objective.
  if (!summary.objective_scalings?.length && typeof summary.objective_scaling === "number") {
    return summary.objective_scaling < 0 ? (names.length ? names : ["every objective"]) : [];
  }
  return (summary.objective_scalings ?? []).flatMap((scaling, i) =>
    typeof scaling === "number" && scaling < 0 ? [names[i] ?? `objective ${i + 1}`] : [],
  );
}

/** Whether a non-empty front has no feasible member (nsga2 then gives the least infeasible one). */
export function frontInfeasible(front: ParetoFront): boolean {
  return front.objectives.length > 0 && front.objectives.every((_, i) => (front.violation[i] ?? 0) > 0);
}

/** One objective's best value on the front, and the member (its index in the front) that has it. */
export interface ObjectiveBest {
  name: string;
  best: number;
  member: number;
}

/**
 * Each objective's best (lowest: all are minimised) value over the front's feasible members,
 * or over all of them when none is feasible; empty for an empty front.
 */
export function objectiveBests(front: ParetoFront): ObjectiveBest[] {
  const members = front.objectives.map((f, i) => ({ f, i }));
  const feasible = members.filter(({ i }) => (front.violation[i] ?? 0) <= 0);
  const pool = feasible.length ? feasible : members;
  if (!pool.length) return [];
  return front.objective_names.map((name, j) => {
    let bestIndex = -1;
    let best = Infinity;
    for (const { f, i } of pool) {
      const v = f[j];
      if (typeof v === "number" && Number.isFinite(v) && (bestIndex < 0 || v < best)) {
        best = v;
        bestIndex = i;
      }
    }
    return { name, best: bestIndex < 0 ? NaN : best, member: bestIndex };
  });
}

/** The hypervolume over the generations (history rows that have one), as chart points. */
export function hypervolumePoints(history: OptimizationStep[]): Point[] {
  return history
    .map((row, i) => ({ row, i }))
    .filter(({ row }) => typeof row.hypervolume === "number")
    .map(({ row, i }) => ({
      x: typeof row.iteration === "number" ? row.iteration : i,
      mean: row.hypervolume as number,
      low: null,
      high: null,
    }));
}

/** A tab of the Examples page: a workflow and its examples (an example may be in several). */
export interface ExampleTab {
  key: string;
  label: string;
  examples: Example[];
}

/**
 * The examples by workflow, in the Plugins page's order of workflows; a workflow that page
 * does not know comes after them, under its own id; empty workflows are left out.
 */
export function examplesByWorkflow(examples: Example[]): ExampleTab[] {
  const known = WORKFLOWS.map((w) => ({ key: w.key, label: w.label }));
  const keys = new Set(known.map((w) => w.key));
  const extra = [...new Set(examples.flatMap((e) => e.entries.map((x) => x.workflow)))]
    .filter((w) => !keys.has(w) && w !== OTHER.key)
    .sort()
    .map((key) => ({ key, label: key.replace(/_/g, " ").replace(/^./, (c) => c.toUpperCase()) }));
  return [...known, ...extra]
    .map(({ key, label }) => ({
      key,
      label,
      examples: examples.filter((e) => e.entries.some((x) => x.workflow === key)),
    }))
    .filter((tab) => tab.examples.length > 0);
}
