import { useEffect, useRef, useState } from "react";

import type { RunSteps, RunSummary } from "../api";
import { fmt, metricLabels, series, stepNumbers } from "../chart";
import { ErrorBox, PathText, Short, Show, Time, useApi, useAsync, useWho } from "../components/common";
import { RunAgents, useAgentList, type AgentList } from "../components/Agents";
import { DeleteButton } from "../components/Delete";
import { MetricChart } from "../components/MetricChart";
import { href } from "../router";
import { annotationItems, isNotFound, noFigures } from "../runs";
import { withoutAdr } from "../text";

export function RunsPage({ initial = "" }: { initial?: string }) {
  const api = useApi();
  const [filter, setFilter] = useState(initial);
  const wanted = Object.fromEntries(
    filter
      .split(",")
      .map((f) => f.trim())
      .filter((f) => f.includes("="))
      .map((f) => f.split("=", 2) as [string, string]),
  );
  const runs = useAsync(() => api.runs(wanted), [JSON.stringify(wanted)]);
  return (
    <section>
      <header className="page-head">
        <h1>Runs</h1>
        <label>
          Annotations{" "}
          <input
            placeholder="study=name, case=m5spots"
            value={filter}
            onChange={(e) => setFilter(e.target.value)}
            size={32}
          />
        </label>
      </header>
      <Show state={runs}>
        {(rows) =>
          rows.length ? (
            <div className="scroll">
              <table>
                <thead>
                  <tr>
                    <th>Run</th>
                    <th>Name</th>
                    <th>Created</th>
                    <th>Annotations</th>
                  </tr>
                </thead>
                <tbody>
                  {[...rows].reverse().map((r) => (
                    <tr key={r.run_id}>
                      <td>
                        <a href={href({ page: "run", id: r.run_id })}>
                          <Short id={r.run_id} />
                        </a>
                      </td>
                      <td>{r.name}</td>
                      <td>
                        <Time iso={r.created} />
                      </td>
                      <td>
                        <ul className="annotations">
                          {annotationItems(r.annotations).map((a) => (
                            <li key={a}>
                              <code>{a}</code>
                            </li>
                          ))}
                        </ul>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : (
            <p className="muted">No runs.</p>
          )
        }
      </Show>
    </section>
  );
}

export function RunPage({ id }: { id: string }) {
  const api = useApi();
  const summary = useAsync(() => api.run(id), [id]);
  const steps = useAsync(() => api.steps(id), [id]);
  // Fetched once for the page: the agents' buttons and the figures' hint both need them.
  const agents = useAgentList();
  // A run that is not there (or hidden: 404) is one message; its steps and figures would
  // only repeat it.
  if (isNotFound(summary.error)) {
    return (
      <section>
        <ErrorBox error={summary.error} />
      </section>
    );
  }
  return (
    <section>
      <Show state={summary}>{(s) => <RunHead id={id} summary={s} />}</Show>
      <RunAgents run={id} names={agents.names} />
      <h2>Performance metrics by step</h2>
      <Show state={steps}>{(s) => <StepCharts steps={s} />}</Show>
      <Figures id={id} agents={agents} />
    </section>
  );
}

function RunHead({ id, summary: s }: { id: string; summary: RunSummary }) {
  return (
    <>
      <header className="page-head">
        <h1>{s.name}</h1>
        <span className="badge">{s.state}</span>
        {s.converged !== null && <span className="badge">{s.converged ? "converged" : "not converged"}</span>}
        <span className="grow" />
        <DeleteButton target={{ run: id }} after={href({ page: "runs" })} />
      </header>
      <dl className="facts">
        <dt>Method</dt>
        <dd>{s.method}</dd>
        <dt>Members</dt>
        <dd>{s.members ?? "–"}</dd>
        <dt>Iterations</dt>
        <dd>{iterationsText(s)}</dd>
        <dt>Posterior</dt>
        <dd>
          iteration {stepText(s, s.posterior_iteration)}
          {s.selected_iteration !== null && ` (selected by cross validation: ${stepText(s, s.selected_iteration)})`}
        </dd>
        <dt>Simulator runs</dt>
        <dd>{s.simulator_runs}</dd>
        <dt>Directory</dt>
        <dd>
          <PathText path={s.path} />
        </dd>
        {Object.entries(s.annotations).map(([k, v]) => (
          <span key={k} className="contents">
            <dt>{k}</dt>
            <dd>{k === "study" ? <a href={href({ page: "study", name: v })}>{v}</a> : v}</dd>
          </span>
        ))}
      </dl>
    </>
  );
}

/** The accepted iterates (0 is the prior), and the rejected trials apart. */
function iterationsText(s: RunSummary): string {
  // An older service gives no accepted_iterations: its records, as before.
  if (s.accepted_iterations === undefined) return s.iterations.join(", ") || "–";
  const accepted = s.accepted_iterations;
  if (accepted.length === 0) return "–";
  const rejected = s.iterations.length - accepted.length;
  const after = `${accepted.length - 1} after the prior`;
  return rejected > 0 ? `${after} (${rejected} rejected trial${rejected === 1 ? "" : "s"})` : after;
}

/** A history record's step number: its place among the accepted iterates. */
function stepText(s: RunSummary, iteration: number | null): string {
  if (iteration === null) return "–";
  const at = (s.accepted_iterations ?? s.iterations).indexOf(iteration);
  return at < 0 ? String(iteration) : String(at);
}

function StepCharts({ steps }: { steps: RunSteps }) {
  const labels = metricLabels(steps.metrics);
  const stepOf = stepNumbers(steps.steps);
  const selected = steps.selected_iteration === null ? null : (stepOf.get(steps.selected_iteration) ?? null);
  const numbered = steps.steps.every((s) => s.label === null);
  return (
    <>
      <p className="muted">
        {steps.method} · {steps.status} · {steps.forecasts} forecasts
      </p>
      {steps.notes.map((n) => (
        <p key={n} className="note">
          {withoutAdr(n)}
        </p>
      ))}
      {numbered && (
        <div className="charts">
          {[...labels].flatMap(([metric, ls]) =>
            ls.map((label) => (
              <MetricChart
                key={`${metric}:${label}`}
                title={label === "all" ? metric : `${metric} · ${label}`}
                points={series(steps.metrics, metric, label, stepOf)}
                selected={selected}
              />
            )),
          )}
        </div>
      )}
      <StepTable steps={steps} labels={labels} />
    </>
  );
}

/** The table view of the charts: every metric's mean ± std per step. */
function StepTable({ steps, labels }: { steps: RunSteps; labels: Map<string, string[]> }) {
  const stepOf = stepNumbers(steps.steps);
  const columns = [...labels].flatMap(([metric, ls]) => ls.map((label) => [metric, label] as const));
  const cell = (metric: string, label: string, iteration: number) =>
    steps.metrics.find((m) => m.metric === metric && m.label === label && m.iteration === iteration);
  return (
    <div className="scroll">
      <table>
        <thead>
          <tr>
            <th>Step</th>
            <th>Members</th>
            <th>Rejected before</th>
            <th>Control</th>
            {columns.map(([metric, label]) => (
              <th key={`${metric}:${label}`}>
                {metric}
                {label !== "all" && <span className="muted"> {label}</span>}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {steps.steps.map((s) => (
            <tr key={s.iteration} className={s.iteration === steps.selected_iteration ? "selected" : ""}>
              <td>
                {s.label ?? stepOf.get(s.iteration)}
                {s.iteration === steps.selected_iteration && " ★"}
              </td>
              <td>{s.n_members}</td>
              <td>{s.rejected_before}</td>
              <td>{fmt(s.control)}</td>
              {columns.map(([metric, label]) => {
                const c = cell(metric, label, s.iteration);
                return (
                  <td key={`${metric}:${label}`} className="num">
                    {c ? (
                      <>
                        {fmt(c.mean)}
                        {c.std !== null && <span className="muted"> ± {fmt(c.std, 2)}</span>}
                      </>
                    ) : (
                      "–"
                    )}
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

/** The run's report figures, fetched with the token and shown from blob URLs. */
function Figures({ id, agents }: { id: string; agents: AgentList }) {
  const api = useApi();
  const { role, failed: whoFailed } = useWho();
  const list = useAsync(() => api.figures(id), [id]);
  const [urls, setUrls] = useState<Record<string, string>>({});
  const [shown, setShown] = useState(12);
  const made = useRef<string[]>([]);
  useEffect(
    () => () => {
      made.current.forEach((u) => URL.revokeObjectURL(u));
      made.current = [];
      setUrls({});
    },
    [id],
  );
  useEffect(() => {
    const wanted = (list.data ?? []).slice(0, shown);
    let live = true;
    for (const path of wanted) {
      if (urls[path]) continue;
      api.artifact(id, path).then((blob) => {
        if (!live) return;
        const url = URL.createObjectURL(blob);
        made.current.push(url);
        setUrls((old) => ({ ...old, [path]: url }));
      }, () => undefined);
    }
    return () => {
      live = false;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [list.data, shown, id]);
  if (!list.data?.length) {
    // Nothing while the figures load: the hint would flash before they appear.
    if (list.data === undefined && list.error === undefined) return null;
    const hint = noFigures(agents.loaded, agents.names.has("plotter"), role, whoFailed);
    if (!hint) return null;
    const { text, command } = hint;
    return (
      <p className="muted">
        {text}
        {command && (
          <>
            {" "}
            <code>{command}</code>
          </>
        )}
      </p>
    );
  }
  return (
    <>
      <h2>Figures ({list.data.length})</h2>
      <div className="figures">
        {list.data.slice(0, shown).map((path) => (
          <figure key={path}>
            {urls[path] ? <img src={urls[path]} alt={path} loading="lazy" /> : <div className="placeholder" />}
            <figcaption>{path.replace(/^report\//, "")}</figcaption>
          </figure>
        ))}
      </div>
      {shown < list.data.length && <button onClick={() => setShown((n) => n + 24)}>Show more</button>}
    </>
  );
}
