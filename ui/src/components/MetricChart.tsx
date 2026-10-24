import { useState } from "react";

import { COLOURS, fmt, scale, type Point } from "../chart";

const W = 520;
const H = 240;
const M = { top: 16, right: 16, bottom: 36, left: 64 };

/**
 * One metric over the steps: the member mean (line and markers) with its p10–p90 band,
 * and the selected step (the posterior) marked. Hovering a step shows its values.
 */
export function MetricChart({
  title,
  points,
  selected,
  xLabel = "iteration",
  caption = "line: member mean; band: p10–p90 over the members",
  valueLabel = "mean",
  restarts = [],
}: {
  title: string;
  points: Point[];
  selected: number | null;
  xLabel?: string;
  /** What the line (and band) are, under the chart when nothing is hovered. */
  caption?: string;
  /** What a point's value is, in the hover readout. */
  valueLabel?: string;
  /** x values where the line starts again rather than joining the previous point. */
  restarts?: number[];
}) {
  const [hover, setHover] = useState<Point | null>(null);
  if (points.length === 0) return null;
  const ys = points.flatMap((p) => [p.mean, p.low ?? p.mean, p.high ?? p.mean]);
  const x = scale(
    points.map((p) => p.x),
    M.left,
    W - M.right,
    false,
  );
  // A little room above and below, so the lowest and highest points sit off the frame.
  const y = scale(ys, H - M.bottom, M.top, true, 0.05);
  const line = points.map((p, i) => `${i && !restarts.includes(p.x) ? "L" : "M"}${x(p.x)},${y(p.mean)}`).join("");
  const banded = points.filter((p) => p.low !== null && p.high !== null);
  const band =
    banded.length > 1
      ? banded.map((p, i) => `${i ? "L" : "M"}${x(p.x)},${y(p.high as number)}`).join("") +
        [...banded]
          .reverse()
          .map((p) => `L${x(p.x)},${y(p.low as number)}`)
          .join("") +
        "Z"
      : null;
  const xTicks = points.length <= 12 ? points.map((p) => p.x) : x.ticks;
  // Enough digits that neighbouring tick labels differ (a narrow range far from zero needs more).
  const digits = y.log ? 2 : tickDigits(y.ticks);
  const step = points.length > 1 ? (W - M.left - M.right) / (points.length - 1) : W;

  return (
    <figure className="chart">
      <figcaption>
        {title}
        {y.log && <span className="muted"> (log scale)</span>}
      </figcaption>
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={`${title} by ${xLabel}`}>
        {y.ticks.map((t) => (
          <g key={t}>
            <line x1={M.left} x2={W - M.right} y1={y(t)} y2={y(t)} stroke={COLOURS.grid} />
            <text x={M.left - 6} y={y(t)} dy="0.32em" textAnchor="end" className="tick">
              {fmt(t, digits)}
            </text>
          </g>
        ))}
        <line x1={M.left} x2={W - M.right} y1={H - M.bottom} y2={H - M.bottom} stroke={COLOURS.baseline} />
        {xTicks.map((t) => (
          <text key={t} x={x(t)} y={H - M.bottom + 16} textAnchor="middle" className="tick">
            {t}
          </text>
        ))}
        <text x={(M.left + W - M.right) / 2} y={H - 4} textAnchor="middle" className="axis-label">
          {xLabel}
        </text>
        {selected !== null && points.some((p) => p.x === selected) && (
          <g>
            <line
              x1={x(selected)}
              x2={x(selected)}
              y1={M.top}
              y2={H - M.bottom}
              stroke={COLOURS.muted}
              strokeDasharray="4 3"
            />
            <text x={x(selected) - 4} y={M.top + 10} textAnchor="end" className="tick">
              selected
            </text>
          </g>
        )}
        {band && <path d={band} fill={COLOURS.posterior} opacity={0.16} />}
        <path d={line} fill="none" stroke={COLOURS.posterior} strokeWidth={2} />
        {points.map((p) => (
          <circle
            key={p.x}
            cx={x(p.x)}
            cy={y(p.mean)}
            r={hover?.x === p.x ? 5.5 : 4}
            fill={COLOURS.posterior}
            stroke="var(--surface)"
            strokeWidth={2}
          />
        ))}
        {points.map((p) => (
          <rect
            key={`hit-${p.x}`}
            x={x(p.x) - step / 2}
            y={M.top}
            width={step}
            height={H - M.top - M.bottom}
            fill="transparent"
            onMouseEnter={() => setHover(p)}
            onMouseLeave={() => setHover(null)}
          />
        ))}
        {hover && (
          <g pointerEvents="none">
            <line x1={x(hover.x)} x2={x(hover.x)} y1={M.top} y2={H - M.bottom} stroke={COLOURS.baseline} />
          </g>
        )}
      </svg>
      <div className="chart-readout" aria-live="polite">
        {hover ? (
          <>
            {xLabel} {hover.x}: {valueLabel} <strong>{fmt(hover.mean, 4)}</strong>
            {hover.low !== null && hover.high !== null && (
              <>
                {" "}
                (p10 {fmt(hover.low, 3)}, p90 {fmt(hover.high, 3)})
              </>
            )}
          </>
        ) : (
          <span className="muted">{caption}</span>
        )}
      </div>
    </figure>
  );
}

/**
 * Significant digits for linear-axis tick labels so that neighbouring ticks read differently:
 * 2 for most axes, more when the ticks are close together relative to their size (e.g. -225.7
 * to -225.8); at most 8, so a range narrower than about 1e-7 of its size may still repeat.
 * (A log axis's ticks are powers of ten and keep 2.)
 */
export function tickDigits(ticks: number[]): number {
  if (ticks.length < 2) return 2;
  const step = Math.abs((ticks[1] as number) - (ticks[0] as number));
  const largest = Math.max(...ticks.map(Math.abs));
  if (!(step > 0) || !(largest > 0)) return 2;
  return Math.min(8, Math.max(2, Math.ceil(Math.log10(largest / step)) + 1));
}
