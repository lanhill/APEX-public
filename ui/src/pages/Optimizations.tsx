import { useCallback } from "react";

import type { OptimizationRow, OptimizationSummary } from "../api";
import { fmt } from "../chart";
import { OptimizationAgents } from "../components/Agents";
import { DeleteButton } from "../components/Delete";
import { PathText, Short, Show, StateBadge, Time, useApi, useAsync, usePolling } from "../components/common";
import { FrontChart } from "../components/FrontChart";
import { MetricChart } from "../components/MetricChart";
import {
  frontInfeasible,
  frontMembers,
  hypervolumePoints,
  isMultiObjective,
  maximisedObjectives,
  objectiveBests,
} from "../front";
import {
  bestControls,
  feasibility,
  historyColumns,
  historyValue,
  objectiveChart,
  objectiveSense,
  optimizationLabel,
  objectiveQuantity,
  plainNumber,
} from "../optimization";
import { href } from "../router";
import { withoutAdr } from "../text";

/** Every optimization job the signed-in user may see, newest first. */
export function OptimizationsPage() {
  const api = useApi();
  const rows = useAsync(() => api.optimizations(), []);
  const active = (rows.data ?? []).some((r) => r.state === "queued" || r.state === "running");
  usePolling(rows.reload, 5000, active);
  return (
    <section>
      <h1>Optimizations</h1>
      <p className="muted">
        Ensemble optimizations run by the service. Submit one from the Editor: a file with{" "}
        <code>[optimization]</code> and <code>[optimizer]</code> tables. Stop or resume it from its job.
      </p>
      <Show state={rows}>{(all) => <OptimizationTable rows={all} />}</Show>
    </section>
  );
}

function OptimizationTable({ rows }: { rows: OptimizationRow[] }) {
  if (!rows.length) return <p className="muted">No optimization yet.</p>;
  const newest = [...rows].sort((a, b) => (b.created ?? "").localeCompare(a.created ?? ""));
  return (
    <div className="scroll">
      <table>
        <thead>
          <tr>
            <th>Optimization</th>
            <th>State</th>
            <th>Created</th>
            <th>Finished</th>
            <th>Job</th>
          </tr>
        </thead>
        <tbody>
          {newest.map((r) => (
            <tr key={r.job}>
              <td>
                {r.path ? (
                  <a href={href({ page: "optimization", path: r.path })} title={r.path}>
                    {optimizationLabel(r.path)}
                  </a>
                ) : (
                  "–"
                )}
                {r.kind === "optimization_resume" && <span className="muted"> (resumed)</span>}
              </td>
              <td>
                <StateBadge state={r.state} />
              </td>
              <td>
                <Time iso={r.created} />
              </td>
              <td>
                <Time iso={r.finished} />
              </td>
              <td>
                <a href={href({ page: "job", id: r.job })}>
                  <Short id={r.job} />
                </a>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

/** One optimization: its facts, objective over the steps, best controls, history and agents. */
export function OptimizationPage({ path }: { path: string }) {
  const api = useApi();
  const summary = useAsync(() => api.optimizationSummary(path), [path]);
  const jobs = useAsync(() => api.optimizations(), []);
  // The newest job of this directory: its page has stop (while running) and resume.
  const job = [...(jobs.data ?? [])]
    .filter((r) => r.path === path)
    .sort((a, b) => (b.created ?? "").localeCompare(a.created ?? ""))[0];
  // Refresh while the run holds its lock, or while its newest job is queued or running (a
  // queued job or resume has no lock yet); the jobs too, so the link follows a resume.
  const active =
    summary.data?.state === "running" || job?.state === "queued" || job?.state === "running";
  const { reload: reloadSummary } = summary;
  const { reload: reloadJobs } = jobs;
  const reload = useCallback(() => {
    reloadSummary();
    reloadJobs();
  }, [reloadSummary, reloadJobs]);
  usePolling(reload, 5000, active);
  return (
    <section>
      <p>
        <a href={href({ page: "optimizations" })}>Back to Optimizations</a>
      </p>
      <Show state={summary}>{(s) => <OptimizationView summary={s} job={job} />}</Show>
      <OptimizationAgents path={path} />
    </section>
  );
}

const SUMMARY_STATE: Record<OptimizationSummary["state"], string> = {
  new: "not started",
  running: "running",
  stopped: "stopped",
  finished: "finished",
};

function OptimizationView({ summary: s, job }: { summary: OptimizationSummary; job: OptimizationRow | undefined }) {
  const chart = objectiveChart(s.history, s.penalised);
  // Several objectives: no single value or line; the front, its hypervolume and
  // each objective's best are shown instead.
  const multi = isMultiObjective(s);
  const feasible = feasibility(s);
  const controls = bestControls(s);
  const columns = historyColumns(s.history);
  const last = chart.points[chart.points.length - 1];
  // Every optimizer minimises: a quantity to maximise, such as NPV, is given a
  // negative scaling. The service says the scaling; an older one does not, and then the page
  // can only guess from all-negative values.
  const sense = objectiveSense(s);
  const value = typeof s.result?.objective === "number" ? s.result.objective : last?.mean;
  const quantity = objectiveQuantity(value, sense, chart.penalised);
  const negative =
    sense.direction === "unknown" && chart.points.length > 0 && chart.points.every((p) => p.mean < 0);
  const scaling = sense.scaling === null ? "" : plainNumber(sense.scaling);
  const scaled =
    sense.scaling === null || sense.scaling === 1
      ? s.objective
      : `${s.objective} ÷ ${sense.scaling < 0 ? `(${scaling})` : scaling}`;
  // Kept in order, a missing value shown as "–", so each realization keeps its number.
  const perRealization = Array.isArray(s.result?.objective_per_realization)
    ? (s.result.objective_per_realization as unknown[]).map((v) => (typeof v === "number" ? v : null))
    : [];
  return (
    <>
      <header className="page-head">
        <h1>{s.name}</h1>
        <span className="badge">run: {SUMMARY_STATE[s.state]}</span>
        {job && (
          <>
            <span className="muted">job:</span>
            <StateBadge state={job.state} />
            <a href={href({ page: "job", id: job.job })}>
              job <Short id={job.job} />
            </a>
          </>
        )}
        <span className="grow" />
        {/* Deleting one's own finished optimization: the preview says what goes
            (for a study's cell, the whole study). */}
        {job && !["queued", "running"].includes(job.state) && (
          <DeleteButton target={{ job: job.job }} after={href({ page: "optimizations" })} />
        )}
      </header>
      <dl className="facts">
        <dt>Optimizer</dt>
        <dd>
          <code>{s.optimizer}</code>
        </dd>
        {multi ? (
          <MultiObjectiveFacts summary={s} />
        ) : (
          <>
            <dt>Objective</dt>
            <dd>
              <code>{s.objective}</code>, {sense.direction === "maximised" ? "maximised" : "minimised"}
              {value !== undefined && (
                <>
                  {" "}
                  · {s.result ? "final" : "latest"} value {fmt(value, 6)}
                  {(chart.penalised || quantity !== null) && (
                    <>
                      {" "}
                      ({[chart.penalised ? "penalised" : null, quantity !== null ? `${s.objective} ${plainNumber(quantity)}` : null]
                        .filter(Boolean)
                        .join("; ")}
                      )
                    </>
                  )}
                </>
              )}
              {scaled !== s.objective && (
                <div className="muted">Every optimizer minimises: here {scaled}.</div>
              )}
              {s.simulated && sense.direction === "minimised" && (
                <div className="muted">
                  If this is a quantity to maximise, such as NPV, give its objective a negative obj_scaling.
                </div>
              )}
              {negative && (
                <div className="muted">
                  All values are negative: if this objective is a quantity to maximise (e.g. NPV), it is minimised as its
                  negative, so lower is better.
                </div>
              )}
            </dd>
          </>
        )}
        <dt>Controls</dt>
        <dd>{s.controls.length}</dd>
        <dt>Simulated evaluations</dt>
        <dd>
          {s.simulated === false ? (
            "none: an analytic objective"
          ) : (
            <>
              {s.evaluations}
              {s.failed_evaluations > 0 && <span className="error-text"> · {s.failed_evaluations} failed</span>}
            </>
          )}
        </dd>
        {s.result?.message && (
          <>
            <dt>Ended because</dt>
            <dd>{withoutAdr(s.result.message)}</dd>
          </>
        )}
        {feasible && (
          <>
            <dt>Constraints</dt>
            <dd>
              {feasible.feasible ? (
                <span className="badge">met</span>
              ) : (
                <span className="badge state-failed">not met</span>
              )}
              {feasible.violation !== null && <> · violation {fmt(feasible.violation, 4)}</>}
              {!feasible.feasible && (
                <div className="muted">
                  The {feasible.of === "result" ? "result" : "latest point"} violates its constraints: its objective is
                  not a valid answer.
                </div>
              )}
            </dd>
          </>
        )}
        <dt>Directory</dt>
        <dd>
          <PathText path={s.path} />
        </dd>
      </dl>
      {multi ? (
        !s.pareto ? (
          <p className="muted">No front yet: it is written at the first checkpoint.</p>
        ) : (
          s.pareto.objectives.length === 0 && (
            <p className="muted">The front is empty: the run has written no member yet.</p>
          )
        )
      ) : chart.points.length > 0 ? (
        <div className="card chart-card">
          <MetricChart
            title={`Objective (${scaled}, minimised${chart.penalised ? ", penalised" : ""})`}
            points={chart.points}
            selected={null}
            restarts={chart.restarts}
            xLabel="history row"
            valueLabel="objective"
            caption={
              (chart.penalised
                ? "at the start, after each accepted step and at each penalty update (EPF), where the line restarts"
                : "at the start (row 0) and after each accepted step") +
              (sense.direction === "maximised" ? `; lower is better (higher ${s.objective})` : "; lower is better")
            }
          />
        </div>
      ) : (
        <p className="muted">No step yet.</p>
      )}
      {s.pareto && s.pareto.objectives.length > 0 && <ParetoCards summary={s} />}
      {controls.length > 0 && (
        <div className="card narrow-card">
          <h2>{multi ? "Representative" : s.result ? "Result" : "Current point"}</h2>
          <div className="scroll">
            <table>
              <thead>
                <tr>
                  <th>Control</th>
                  <th>Value</th>
                </tr>
              </thead>
              <tbody>
                {controls.map((c) => (
                  <tr key={c.name}>
                    <td>
                      <code>{c.name}</code>
                    </td>
                    <td className="num">{fmt(c.value, 6)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
      {/* A multi-objective result's per-realization values are its representative's first
          objective only: not shown. */}
      {!multi && perRealization.length > 1 && (
        <div className="card narrow-card">
          <h2>Objective per realization</h2>
          <div className="scroll">
            <table>
              <thead>
                <tr>
                  <th>Realization</th>
                  <th>Objective</th>
                </tr>
              </thead>
              <tbody>
                {perRealization.map((v, i) => (
                  <tr key={i}>
                    <td>{i}</td>
                    <td className="num">{v === null ? "–" : fmt(v, 6)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
      {s.history.length > 0 && (
        <details className="card">
          <summary>History ({s.history.length} rows)</summary>
          <div className="scroll">
            <table>
              <thead>
                <tr>
                  {columns.map((c) => (
                    <th key={c}>{c}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {s.history.map((row, i) => (
                  <tr key={i}>
                    {columns.map((c) => (
                      <td key={c} className={typeof row[c] === "number" ? "num" : undefined}>
                        {historyValue(row[c], fmt)}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </details>
      )}
    </>
  );
}

/**
 * A multi-objective run's facts: its objectives, the front's size and hypervolume,
 * and each objective's best on the front. No single value: the representative's objectives
 * are in the front's table.
 */
function MultiObjectiveFacts({ summary: s }: { summary: OptimizationSummary }) {
  const front = s.pareto;
  const bests = front && front.objectives.length > 0 ? objectiveBests(front) : [];
  // No feasible member yet: the front is the least infeasible member, not an answer.
  const infeasible = front ? frontInfeasible(front) : false;
  const names = front?.objective_names ?? s.objective_names ?? [];
  const maximised = maximisedObjectives(s);
  return (
    <>
      <dt>Objectives</dt>
      <dd>
        {names.length > 0 ? (
          <>
            {names.length} objectives, all minimised:{" "}
            {names.map((n, i) => (
              <span key={n}>
                {i > 0 && ", "}
                <code>{n}</code>
              </span>
            ))}
          </>
        ) : (
          <>several objectives, all minimised</>
        )}
        {maximised.length > 0 && (
          <div className="muted">
            Every optimizer minimises: {maximised.join(", ")} {maximised.length === 1 ? "is" : "are"} maximised, so the
            values here are scaled by a negative obj_scaling (minimised as negatives), and lower is better.
          </div>
        )}
        <div className="muted">
          A multi-objective run has no single objective value: this page shows its front, the front's hypervolume and
          each objective's best on the front. Its single answer is the representative, the member closest to the
          ideal point.
        </div>
      </dd>
      {front && (
        <>
          <dt>Front</dt>
          <dd>
            {front.objectives.length} members · hypervolume {fmt(front.hypervolume, 4)}
          </dd>
          {bests.length > 0 && (
            <>
              <dt>Best on the front</dt>
              <dd>
                {infeasible && (
                  <div>
                    <span className="badge state-failed">infeasible</span>{" "}
                    <span className="muted">
                      No member is feasible yet: the front is the least infeasible member, and these values are not
                      a valid answer.
                    </span>
                  </div>
                )}
                {bests.map((b, i) => (
                  <span key={b.name}>
                    {i > 0 && " · "}
                    <code>{b.name}</code> {fmt(b.best, 6)}
                    {b.member >= 0 && <span className="muted"> (member {b.member + 1})</span>}
                  </span>
                ))}
              </dd>
            </>
          )}
        </>
      )}
    </>
  );
}

/** A multi-objective run's front, its hypervolume over the generations and its members. */
function ParetoCards({ summary: s }: { summary: OptimizationSummary }) {
  const front = s.pareto;
  if (!front) return null;
  const hv = hypervolumePoints(s.history);
  const members = frontMembers(front);
  return (
    <>
      <div className="card chart-card">
        <FrontChart front={front} />
      </div>
      {hv.length > 0 && (
        <div className="card chart-card">
          <MetricChart
            title="Hypervolume of the front"
            points={hv}
            selected={null}
            xLabel="generation"
            valueLabel="hypervolume"
            caption="the space the front dominates up to the run's reference point: higher is better"
          />
        </div>
      )}
      <details className="card">
        <summary>Front members ({members.length})</summary>
        <div className="scroll">
          <table>
            <thead>
              <tr>
                <th>#</th>
                {front.objective_names.map((n) => (
                  <th key={n}>{n}</th>
                ))}
                {s.controls.map((c) => (
                  <th key={c}>
                    <code>{c}</code>
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {members.map((m) => (
                <tr key={m.index} className={m.representative ? "selected" : undefined}>
                  <td>
                    {m.index + 1}
                    {m.representative && <span className="badge">representative</span>}
                    {!m.feasible && <span className="badge state-failed">infeasible</span>}
                  </td>
                  {m.objectives.map((v, j) => (
                    <td key={j} className="num">
                      {fmt(v, 5)}
                    </td>
                  ))}
                  {m.x.map((v, j) => (
                    <td key={`x${j}`} className="num">
                      {fmt(v, 5)}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </details>
    </>
  );
}

