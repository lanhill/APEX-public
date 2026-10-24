// A multi-objective run's Pareto front: two objectives as a scatter, three or more
// as parallel coordinates; the representative (the run's single answer) is marked.

import { useState } from "react";

import type { ParetoFront } from "../api";
import { COLOURS, fmt, scale } from "../chart";
import { frontMembers, parallelRows, type FrontMember } from "../front";

const W = 640;
const H = 340;
const M = { top: 16, right: 18, bottom: 44, left: 64 };

export function FrontChart({ front }: { front: ParetoFront }) {
  const [hover, setHover] = useState<FrontMember | null>(null);
  const members = frontMembers(front);
  const names = front.objective_names;
  const shown = hover ?? members.find((m) => m.representative) ?? null;
  return (
    <figure className="chart">
      <figcaption>
        Pareto front ({members.length} members, means over realizations, all minimised)
      </figcaption>
      {names.length === 2 ? (
        <Scatter members={members} names={names} hover={hover} onHover={setHover} />
      ) : (
        <Parallel front={front} members={members} names={names} hover={hover} onHover={setHover} />
      )}
      <p className="muted small">
        {shown
          ? `${shown.representative ? "Representative" : `Member ${shown.index + 1}`}: ` +
            names.map((n, j) => `${n} ${fmt(shown.objectives[j], 4)}`).join(", ") +
            (shown.feasible ? "" : " (infeasible)")
          : "Hover a member to read its objectives."}
        {" "}The representative (ringed) is the member closest to the ideal point in normalised objectives.
      </p>
    </figure>
  );
}

function Scatter({
  members,
  names,
  hover,
  onHover,
}: {
  members: FrontMember[];
  names: string[];
  hover: FrontMember | null;
  onHover: (m: FrontMember | null) => void;
}) {
  const x = scale(members.map((m) => m.objectives[0] ?? 0), M.left, W - M.right, false, 0.05);
  const y = scale(members.map((m) => m.objectives[1] ?? 0), H - M.bottom, M.top, false, 0.05);
  return (
    <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={`Pareto front: ${names[1]} against ${names[0]}`}>
      {y.ticks.map((t) => (
        <g key={`y${t}`}>
          <line x1={M.left} x2={W - M.right} y1={y(t)} y2={y(t)} stroke={COLOURS.grid} />
          <text x={M.left - 6} y={y(t)} dy="0.32em" textAnchor="end" className="tick">
            {fmt(t, 3)}
          </text>
        </g>
      ))}
      {x.ticks.map((t) => (
        <text key={`x${t}`} x={x(t)} y={H - M.bottom + 16} textAnchor="middle" className="tick">
          {fmt(t, 3)}
        </text>
      ))}
      <line x1={M.left} x2={W - M.right} y1={H - M.bottom} y2={H - M.bottom} stroke={COLOURS.baseline} />
      <text x={(M.left + W - M.right) / 2} y={H - 6} textAnchor="middle" className="axis-label">
        {names[0]}
      </text>
      <text x={14} y={(M.top + H - M.bottom) / 2} textAnchor="middle" className="axis-label"
        transform={`rotate(-90 14 ${(M.top + H - M.bottom) / 2})`}>
        {names[1]}
      </text>
      {members.map((m) => (
        <circle
          key={m.index}
          cx={x(m.objectives[0] ?? 0)}
          cy={y(m.objectives[1] ?? 0)}
          r={hover?.index === m.index ? 6 : 4}
          fill={m.feasible ? COLOURS.posterior : COLOURS.muted}
          stroke="var(--surface)"
          strokeWidth={1.5}
          onMouseEnter={() => onHover(m)}
          onMouseLeave={() => onHover(null)}
        />
      ))}
      {members
        .filter((m) => m.representative)
        .map((m) => (
          <circle key="rep" cx={x(m.objectives[0] ?? 0)} cy={y(m.objectives[1] ?? 0)} r={9} fill="none"
            stroke={COLOURS.prior} strokeWidth={2.5} pointerEvents="none" />
        ))}
    </svg>
  );
}

function Parallel({
  front,
  members,
  names,
  hover,
  onHover,
}: {
  front: ParetoFront;
  members: FrontMember[];
  names: string[];
  hover: FrontMember | null;
  onHover: (m: FrontMember | null) => void;
}) {
  const rows = parallelRows(front);
  const step = (W - M.left - M.right) / Math.max(names.length - 1, 1);
  const xAt = (j: number) => M.left + j * step;
  const yAt = (v: number) => H - M.bottom - v * (H - M.top - M.bottom);
  const path = (row: number[]) => row.map((v, j) => `${j ? "L" : "M"}${xAt(j)},${yAt(v)}`).join(" ");
  const ordered = [...members].sort((a, b) => Number(a.representative) - Number(b.representative));
  return (
    <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={`Pareto front over ${names.join(", ")}`}>
      {names.map((n, j) => (
        <g key={n}>
          <line x1={xAt(j)} x2={xAt(j)} y1={M.top} y2={H - M.bottom} stroke={COLOURS.baseline} />
          <text x={xAt(j)} y={H - M.bottom + 18} textAnchor="middle" className="axis-label">
            {n}
          </text>
        </g>
      ))}
      <text x={M.left - 8} y={M.top} dy="0.32em" textAnchor="end" className="tick">worst</text>
      <text x={M.left - 8} y={H - M.bottom} dy="0.32em" textAnchor="end" className="tick">best</text>
      {ordered.map((m) => (
        <path
          key={m.index}
          d={path(rows[m.index] ?? [])}
          fill="none"
          stroke={m.representative ? COLOURS.prior : COLOURS.posterior}
          strokeWidth={m.representative || hover?.index === m.index ? 2.5 : 1}
          opacity={m.representative || hover?.index === m.index ? 1 : 0.4}
          onMouseEnter={() => onHover(m)}
          onMouseLeave={() => onHover(null)}
        />
      ))}
    </svg>
  );
}
