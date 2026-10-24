// The Optimizations pages' model: an optimization's objective over its steps, its
// best controls and its history as a table. Display only: every number comes from
// GET /v1/optimizations/summary.

import type { Point } from "./chart";
import type { OptimizationStep, OptimizationSummary } from "./api";

export interface ObjectiveChart {
  /** The objective at each history row that has one, in order: x is the row's place (0: the start). */
  points: Point[];
  /** Where an EPF penalty update starts the line again: its value is penalised anew. */
  restarts: number[];
  /** Whether the run has penalty updates (EPF), so the values are penalised objectives. */
  penalised: boolean;
}

/**
 * The objective over the history. Rows are a start, accepted steps and, under EPF, penalty
 * updates (`epf_update`), which reset `iteration` and re-penalise the objective; so the x axis
 * is the row's place, and the line restarts at each update rather than drawing a jump.
 */
export function objectiveChart(history: OptimizationStep[], configured?: boolean): ObjectiveChart {
  const rows = history.filter((row) => typeof row.objective === "number" && Number.isFinite(row.objective));
  const points = rows.map((row, i) => ({ x: i, mean: row.objective as number, low: null, high: null }));
  const restarts = rows.flatMap((row, i) => (row.kind === "epf_update" ? [i] : []));
  // The service says whether EPF is configured (its penalty is in every value, from the first
  // outer iteration); an older one does not, and then only an update row shows it.
  return { points, restarts, penalised: configured ?? history.some((row) => row.kind === "epf_update") };
}

/**
 * The controls to show, with their names: the result's once finished, else the last accepted
 * step's, else the start's. Empty before the first row.
 */
export function bestControls(summary: OptimizationSummary): { name: string; value: number }[] {
  const fromResult = summary.result?.x;
  const latest = (kind: string) =>
    [...summary.history].reverse().find((row) => row.kind === kind && Array.isArray(row.x))?.x;
  const x = fromResult ?? latest("step") ?? latest("start") ?? [];
  const names = (fromResult && summary.result?.names) || summary.controls;
  return x.map((value, i) => ({ name: names[i] ?? `x[${i}]`, value }));
}

/** History columns shared by the optimizers first, then each optimizer's own, as they appear. */
const FIRST = ["iteration", "kind", "objective", "nfev", "njev", "member_evaluations", "epf_iteration"];

export function historyColumns(history: OptimizationStep[]): string[] {
  const seen = new Set<string>();
  for (const row of history) for (const key of Object.keys(row)) if (key !== "x") seen.add(key);
  return [...FIRST.filter((k) => seen.has(k)), ...[...seen].filter((k) => !FIRST.includes(k))];
}

/** An optimization's short name from its directory: `.../optimizations/<name>/<id>` → `<name>/<id>`. */
export function optimizationLabel(path: string | null): string {
  if (!path) return "–";
  const parts = path.split("/").filter(Boolean);
  return parts.slice(-2).join("/") || path;
}

/** A history value as text: counters as integers, other numbers to 6 significant digits. */
export function historyValue(value: unknown, fmt: (v: number, digits: number) => string): string {
  if (value === undefined || value === null) return "–";
  if (typeof value === "number") return Number.isInteger(value) ? String(value) : fmt(value, 6);
  return String(value);
}

/**
 * Which way the objective's quantity goes. Every optimizer minimises the objective, which is
 * the quantity divided by its `obj_scaling` (PET's convention): a negative scaling maximises
 * the quantity (e.g. NPV), a positive one or none minimises it. "unknown" for a service that
 * does not send the scaling.
 */
export interface ObjectiveSense {
  direction: "maximised" | "minimised" | "unknown";
  /** The obj_scaling, when there is one. */
  scaling: number | null;
}

export function objectiveSense(summary: OptimizationSummary): ObjectiveSense {
  const scaling = summary.objective_scaling;
  if (scaling === undefined) return { direction: "unknown", scaling: null };
  if (scaling === null) return { direction: "minimised", scaling: null };
  return { direction: scaling < 0 ? "maximised" : "minimised", scaling };
}

/** The quantity behind a minimised value: value × obj_scaling (e.g. NPV from its scaled negative). */
export function unscaled(value: number, sense: ObjectiveSense): number | null {
  return sense.scaling === null ? null : value * sense.scaling;
}

/**
 * The quantity to show beside a value (e.g. NPV), or null: none without a scaling other than 1,
 * and none when the value is penalised. EPF adds its penalty to the value being minimised, so
 * unscaling it would not give the quantity (`penalised`: the service's own flag, or with an
 * older service the first `epf_update` row, from a run's second outer iteration on).
 */
export function objectiveQuantity(value: number | undefined, sense: ObjectiveSense, penalised: boolean): number | null {
  if (value === undefined || penalised || sense.scaling === null || sense.scaling === 1) return null;
  return unscaled(value, sense);
}

/**
 * A number as a person reads it, grouped and without exponents: 11200000 → "11,200,000". The
 * locale is fixed ("en"): "," as the grouping mark reads unambiguously next to "÷" and the
 * other numbers of the page (`fmt`'s), whatever the browser's locale.
 */
export function plainNumber(value: number, digits = 6): string {
  return new Intl.NumberFormat("en", { maximumSignificantDigits: digits }).format(value);
}

/** A recorded `feasible`: a number (the backend's 1.0/0.0) or a boolean; anything else is unknown. */
function feasibleOf(value: unknown): boolean | null {
  if (typeof value === "boolean") return value;
  if (typeof value === "number" && Number.isFinite(value)) return value !== 0;
  return null;
}

/**
 * Whether the result (or, before it, the latest row) meets its constraints: a constrained run
 * (e.g. ropt with `constraints = true`) records `feasible` (1/0) and `violation` in its history
 * rows and in result.json, where the optimizer's extras sit at the top level (`result_summary`;
 * an `extra` table is read too). Null when nothing usable is recorded (no constraints).
 */
export function feasibility(
  summary: OptimizationSummary,
): { feasible: boolean; violation: number | null; of: "result" | "latest" } | null {
  const result = summary.result as Record<string, unknown> | null;
  const extra = result && typeof result.extra === "object" && result.extra ? (result.extra as Record<string, unknown>) : null;
  const sources: [Record<string, unknown> | null | undefined, "result" | "latest"][] = [
    [result, "result"],
    [extra, "result"],
    [[...summary.history].reverse().find((r) => feasibleOf(r.feasible) !== null), "latest"],
  ];
  for (const [row, of] of sources) {
    const feasible = row ? feasibleOf(row.feasible) : null;
    if (row && feasible !== null) {
      const violation = typeof row.violation === "number" && Number.isFinite(row.violation) ? row.violation : null;
      return { feasible, violation, of };
    }
  }
  return null;
}
