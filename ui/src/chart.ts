// Scales and series for the metric charts (pure functions, so they are unit-tested).

import type { StepStatistics } from "./api";

// The validated palette of src/apex/reporting/plots.py (posterior blue, prior orange).
export const COLOURS = {
  posterior: "#2a78d6",
  prior: "#eb6834",
  ink: "#0b0b0b",
  secondary: "#52514e",
  muted: "#898781",
  grid: "#e1e0d9",
  baseline: "#c3c2b7",
} as const;

export interface Point {
  x: number;
  mean: number;
  low: number | null;
  high: number | null;
}

/**
 * One metric and label over the steps, in step order; missing means are left out. With
 * ``stepOf`` (history record -> step number), x is the step number and rows of other
 * records (rejected trials) are left out.
 */
export function series(
  rows: StepStatistics[],
  metric: string,
  label: string,
  stepOf?: Map<number, number>,
): Point[] {
  return rows
    .filter((r) => r.metric === metric && r.label === label && r.mean !== null)
    .filter((r) => stepOf === undefined || stepOf.has(r.iteration))
    .sort((a, b) => a.iteration - b.iteration)
    .map((r) => ({
      x: stepOf?.get(r.iteration) ?? r.iteration,
      mean: r.mean as number,
      low: r.p10,
      high: r.p90,
    }));
}

/** History record -> step number (0, 1, 2, ...: rejected trials not counted). */
export function stepNumbers(steps: { iteration: number; step?: number }[]): Map<number, number> {
  return new Map(steps.map((s, i) => [s.iteration, s.step ?? i]));
}

/**
 * The data mismatches (data_mismatch, normalized_data_mismatch and their cv_/test_
 * versions) are shown overall only, not per data type.
 */
export function shownLabel(metric: string, label: string): boolean {
  return !metric.endsWith("data_mismatch") || label === "all";
}

/** The metrics and their labels to show, in the order first seen. */
export function metricLabels(rows: StepStatistics[]): Map<string, string[]> {
  const found = new Map<string, string[]>();
  for (const row of rows) {
    if (!shownLabel(row.metric, row.label)) continue;
    const labels = found.get(row.metric) ?? [];
    if (!labels.includes(row.label)) labels.push(row.label);
    found.set(row.metric, labels);
  }
  return found;
}

export interface Scale {
  (value: number): number;
  ticks: number[];
  log: boolean;
}

/**
 * A linear or log10 scale from [min, max] of `values` onto [from, to].
 * Log is used when every value is positive and they span more than two decades,
 * as data mismatches usually do. `pad` widens the range by that fraction of it at each end
 * (in log space for a log scale), so the extreme values are not drawn on the frame; a negative
 * one counts as 0.
 */
export function scale(values: number[], from: number, to: number, allowLog = true, pad = 0): Scale {
  const finite = values.filter(Number.isFinite);
  let lo = finite.length ? Math.min(...finite) : 0;
  let hi = finite.length ? Math.max(...finite) : 1;
  const log = allowLog && lo > 0 && hi / lo > 100;
  const f = log ? Math.log10 : (v: number) => v;
  if (lo === hi) {
    const half = lo === 0 ? 1 : Math.abs(lo) * 0.1;
    lo -= half;
    hi += half;
  }
  const margin = (f(hi) - f(lo)) * Math.max(pad, 0);
  const a = f(lo) - margin;
  const b = f(hi) + margin;
  const map = ((v: number) => from + ((f(v) - a) / (b - a)) * (to - from)) as Scale;
  map.log = log;
  // Ticks over the visible range: the padded one (only data's decades on a log scale).
  map.ticks = log ? logTicks(lo, hi) : niceTicks(a, b);
  return map;
}

export function niceTicks(lo: number, hi: number, count = 5): number[] {
  const span = hi - lo;
  const raw = span / count;
  const power = 10 ** Math.floor(Math.log10(raw));
  const step = ([1, 2, 5, 10].find((m) => m * power >= raw) ?? 10) * power;
  const ticks: number[] = [];
  for (let t = Math.ceil(lo / step) * step; t <= hi + step * 1e-9; t += step) {
    ticks.push(Number(t.toPrecision(12)));
  }
  return ticks;
}

function logTicks(lo: number, hi: number): number[] {
  const ticks: number[] = [];
  for (let e = Math.ceil(Math.log10(lo)); e <= Math.floor(Math.log10(hi)); e++) ticks.push(10 ** e);
  return ticks;
}

/** A compact number for axes and tables. */
export function fmt(value: number | null | undefined, digits = 3): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return "–";
  const magnitude = Math.abs(value);
  if (magnitude !== 0 && (magnitude >= 1e5 || magnitude < 1e-3)) return value.toExponential(digits - 1);
  return Number(value.toPrecision(digits)).toString();
}
