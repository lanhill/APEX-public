import type { PairedDifference, VariantStats } from "./api";

/**
 * The paired difference of a study's (method, variant) group against a reference group, if
 * the comparison has it. Without methods the methods are null on both sides.
 */
export function difference(
  differences: PairedDifference[],
  v: VariantStats,
  reference: { method: string | null; variant: string },
): PairedDifference | undefined {
  return differences.find(
    (x) =>
      x.variant === v.variant &&
      (x.method ?? null) === (v.method ?? null) &&
      x.reference === reference.variant &&
      (x.reference_method ?? null) === reference.method,
  );
}

/** Where a method ran or runs; older services send no engine, which was APEX's. */
export function engineOf(item: { engine?: string | null }): string {
  return item.engine ?? "apex";
}

/** Whether any item runs on another engine than APEX's, so that an engine column says something. */
export function anyOtherEngine(items: { engine?: string | null }[]): boolean {
  return items.some((item) => engineOf(item) !== "apex");
}

/**
 * One cell of a study, as its annotations describe it: a history-matching run (the catalogue's)
 * or an optimization (a job whose directory is the result).
 */
export interface StudyCellRow {
  /** The run's id, or the optimization job's. */
  key: string;
  kind: "run" | "optimization";
  /** The run's id, or the optimization's directory: where its page is. */
  target: string;
  case: string;
  method: string | null;
  engine: string;
  variant: string;
  replicate: number;
}

type Annotated = { annotations: Record<string, string> };

function cell(a: Annotated["annotations"]): Omit<StudyCellRow, "key" | "kind" | "target"> {
  return {
    case: a.case ?? "",
    method: a.method ?? null,
    engine: a.engine ?? "apex",
    variant: a.variant ?? "",
    replicate: Number(a.replicate ?? 0),
  };
}

/**
 * A study's cells from its runs' and optimizations' annotations (the catalogue marks a run of
 * another engine with `engine`; APEX's carry none), ordered by case, method, variant and
 * replicate. An optimization is listed by its newest job (a resume has the same directory).
 */
export function studyCells(
  runs: ({ run_id: string } & Annotated)[],
  study: string,
  optimizations: ({ job: string; path: string | null; created?: string | null } & Annotated)[] = [],
): StudyCellRow[] {
  const byPath = new Map<string, StudyCellRow & { created: string }>();
  for (const o of optimizations) {
    if (o.annotations.study !== study || !o.path) continue;
    const created = o.created ?? "";
    const seen = byPath.get(o.path);
    if (seen && seen.created >= created) continue;
    byPath.set(o.path, { key: o.job, kind: "optimization", target: o.path, created, ...cell(o.annotations) });
  }
  const fromRuns: StudyCellRow[] = runs
    .filter((r) => r.annotations.study === study)
    .map((r) => ({ key: r.run_id, kind: "run", target: r.run_id, ...cell(r.annotations) }));
  const fromOptimizations: StudyCellRow[] = [...byPath.values()].map(({ created: _, ...row }) => row);
  return [...fromRuns, ...fromOptimizations]
    .sort(
      (a, b) =>
        a.case.localeCompare(b.case) ||
        (a.method ?? "").localeCompare(b.method ?? "") ||
        a.variant.localeCompare(b.variant) ||
        a.replicate - b.replicate,
    );
}

/** `set` with `key` added if absent and removed if present (a new set; React state is immutable). */
export function toggled(set: Set<string>, key: string): Set<string> {
  const next = new Set(set);
  if (next.has(key)) next.delete(key);
  else next.add(key);
  return next;
}

/** The studies one may see, with their runs and optimizations, from the annotations. */
export function studyNames(
  runs: Annotated[],
  optimizations: ({ path: string | null } & Annotated)[],
): { name: string; runs: number; optimizations: number }[] {
  const found = new Map<string, { runs: number; paths: Set<string> }>();
  const entry = (name: string) => found.get(name) ?? found.set(name, { runs: 0, paths: new Set() }).get(name)!;
  for (const r of runs) if (r.annotations.study) entry(r.annotations.study).runs += 1;
  for (const o of optimizations) if (o.annotations.study && o.path) entry(o.annotations.study).paths.add(o.path);
  return [...found].map(([name, f]) => ({ name, runs: f.runs, optimizations: f.paths.size }));
}

/**
 * A comparison metric as a heading: a study of optimizations compares `objective:final`,
 * `objective:best`, `evaluations:all` and `iterations:all`; a run's metrics keep
 * their `metric:label` names.
 */
export function comparisonMetricLabel(metric: string): string {
  const named: Record<string, string> = {
    "objective:final": "objective at the end (minimised)",
    "objective:best": "best objective (minimised)",
    "evaluations:all": "evaluations",
    "iterations:all": "accepted steps",
  };
  return Object.prototype.hasOwnProperty.call(named, metric) ? (named[metric] as string) : metric;
}
