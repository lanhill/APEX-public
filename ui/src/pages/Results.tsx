import { useState } from "react";

import type { Comparison, PairedDifference, PredictionSummary } from "../api";
import {
  anyOtherEngine,
  comparisonMetricLabel,
  difference,
  engineOf,
  studyCells,
  studyNames,
  type StudyCellRow,
} from "../comparison";
import { fmt } from "../chart";
import { ErrorBox, PathText, Short, Show, StateBadge, useApi, useAsync } from "../components/common";
import { StudyReport } from "../components/Agents";
import { DeleteButton } from "../components/Delete";
import { href } from "../router";
import { withoutAdr } from "../text";

/** "3 runs", "2 optimizations", or both. */
function cellCount(runs: number, optimizations: number): string {
  const parts = [];
  if (runs) parts.push(`${runs} run${runs === 1 ? "" : "s"}`);
  if (optimizations) parts.push(`${optimizations} optimization${optimizations === 1 ? "" : "s"}`);
  return parts.join(", ");
}

/** Studies, found by the `study` annotation of the catalogue's runs and of optimizations. */
export function StudiesPage() {
  const api = useApi();
  const runs = useAsync(() => api.runs(), []);
  const optimizations = useAsync(() => api.optimizations(), []);
  const [name, setName] = useState("");
  return (
    <section>
      <header className="page-head">
        <h1>Studies</h1>
        <form
          onSubmit={(e) => {
            e.preventDefault();
            if (name) window.location.hash = href({ page: "study", name });
          }}
        >
          <input placeholder="study name" value={name} onChange={(e) => setName(e.target.value)} />{" "}
          <button type="submit">Open</button>
        </form>
      </header>
      {/* Runs and optimizations load apart: either shows as soon as it comes, and a failure of
          one never hides the other's studies nor claims there are none. */}
      <ErrorBox error={runs.error} />
      <ErrorBox error={optimizations.error} />
      {(() => {
        const studies = studyNames(runs.data ?? [], optimizations.data ?? []);
        const both = runs.data !== undefined && optimizations.data !== undefined;
        const pending = runs.loading || optimizations.loading;
        if (studies.length) {
          return (
            <>
              <ul className="list">
                {studies.map((s) => (
                  <li key={s.name}>
                    <a href={href({ page: "study", name: s.name })}>{s.name}</a>{" "}
                    <span className="muted">{cellCount(s.runs, s.optimizations)}</span>
                  </li>
                ))}
              </ul>
              {pending && <p className="muted">Loading…</p>}
            </>
          );
        }
        if (both) return <p className="muted">No study has runs or optimizations yet.</p>;
        return pending ? <p className="muted">Loading…</p> : null;
      })()}
    </section>
  );
}

export function StudyPage({ name }: { name: string }) {
  const api = useApi();
  const status = useAsync(() => api.study(name), [name]);
  const comparison = useAsync(() => api.comparison(name), [name]);
  const runs = useAsync(() => api.runs({ study: name }), [name]);
  const optimizations = useAsync(() => api.optimizations(), []);
  const cells = studyCells(runs.data ?? [], name, optimizations.data ?? []);
  const hasRuns = cells.some((c) => c.kind === "run");
  return (
    <section>
      <header className="page-head">
        <h1>Study {name}</h1>
        {status.data && <StateBadge state={status.data.state} />}
        {status.data && (
          <a href={href({ page: "job", id: status.data.id })}>
            job <Short id={status.data.id} />
          </a>
        )}
        <span className="grow" />
        {status.data && !["queued", "running"].includes(status.data.state) && (
          <DeleteButton target={{ study: name }} after={href({ page: "studies" })} />
        )}
      </header>
      {status.data && (
        <p className="muted">
          {Object.values(status.data.children).filter((s) => s === "complete").length} of{" "}
          {Object.keys(status.data.children).length} cells complete
          {hasRuns && (
            <>
              {" · "}
              <a href={href({ page: "runs", filter: `study=${name}` })}>runs</a>
            </>
          )}
        </p>
      )}
      {status.data?.state === "complete" && <StudyReport study={name} />}
      <Show state={comparison}>{(c) => <ComparisonView comparison={c} />}</Show>
      <ErrorBox error={runs.error} />
      <ErrorBox error={optimizations.error} />
      <CellsView cells={cells} />
      {!cells.length && (runs.loading || optimizations.loading) && <p className="muted">Loading the cells…</p>}
    </section>
  );
}

function DifferenceCells({ d }: { d: PairedDifference | undefined }) {
  return (
    <>
      <td className="num">{d ? fmt(d.mean_difference) : "–"}</td>
      <td className="num">{d ? fmt(d.standard_error) : "–"}</td>
      <td className="num" title={d?.note && withoutAdr(d.note)}>
        {d ? fmt(d.p_value, 2) : "–"}
      </td>
    </>
  );
}

/**
 * A study's cells, with the engine each runs on: runs from the catalogue, and
 * optimizations, each linked to its own page.
 */
function CellsView({ cells }: { cells: StudyCellRow[] }) {
  if (!cells.length) return null;
  const methods = cells.some((c) => c.method !== null);
  // The engine column only when some cell runs on another engine than APEX's.
  const engines = anyOtherEngine(cells);
  return (
    <div className="card">
      <h3>Cells</h3>
      <div className="scroll">
        <table>
          <thead>
            <tr>
              <th>Case</th>
              {methods && <th>Method</th>}
              {engines && <th>Engine</th>}
              <th>Variant</th>
              <th>Replicate</th>
              <th>Result</th>
            </tr>
          </thead>
          <tbody>
            {cells.map((c) => (
              <tr key={c.key}>
                <td>{c.case}</td>
                {methods && <td>{c.method ?? "–"}</td>}
                {engines && <td>{c.engine}</td>}
                <td>{c.variant}</td>
                <td>{c.replicate}</td>
                <td>
                  {c.kind === "run" ? (
                    <a href={href({ page: "run", id: c.target })}>
                      run <Short id={c.target} />
                    </a>
                  ) : (
                    <a href={href({ page: "optimization", path: c.target })} title={c.target}>
                      optimization
                    </a>
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function ComparisonView({ comparison: c }: { comparison: Comparison }) {
  // With methods: each variant against the first within its method, and each
  // method against the first with the same variant.
  const refMethod = c.reference_method ?? null;
  // The engine column only when some method runs on another engine.
  const engines = anyOtherEngine(c.metrics.flatMap((m) => m.variants));
  return (
    <>
      <p className="muted">
        {c.replicates} replicate{c.replicates === 1 ? "" : "s"} per variant · differences against{" "}
        <strong>{c.reference}</strong>
        {refMethod && (
          <>
            {" "}
            within each method, and against <strong>{refMethod}</strong> with the same variant
          </>
        )}{" "}
        (paired over replicates; Wilcoxon signed-rank p-values)
      </p>
      {c.missing.length > 0 && <p className="note">Without results yet: {withoutAdr(c.missing.join(", "))}</p>}
      {(c.skipped ?? []).map((s) => (
        <p key={s.label} className="note">
          Not run: {s.label}: {withoutAdr(s.reason)}
        </p>
      ))}
      {c.notes.map((n) => (
        <p key={n} className="note">
          {withoutAdr(n)}
        </p>
      ))}
      {c.metrics.map((m) => (
        <div key={`${m.case}:${m.metric}`} className="card">
          <h3>
            {m.case} · {comparisonMetricLabel(m.metric)}
          </h3>
          <div className="scroll">
            <table>
              <thead>
                <tr>
                  {refMethod && <th>Method</th>}
                  {engines && <th>Engine</th>}
                  <th>Variant</th>
                  <th>n</th>
                  <th>Mean</th>
                  <th>Std</th>
                  <th>Min</th>
                  <th>Max</th>
                  <th>Δ vs {c.reference}</th>
                  <th>SE</th>
                  <th>p</th>
                  {refMethod && (
                    <>
                      <th>Δ vs {refMethod}</th>
                      <th>SE</th>
                      <th>p</th>
                    </>
                  )}
                </tr>
              </thead>
              <tbody>
                {m.variants.map((v) => {
                  const method = v.method ?? null;
                  const d = difference(m.differences, v, { method, variant: c.reference });
                  const dm = refMethod
                    ? difference(m.differences, v, { method: refMethod, variant: v.variant })
                    : undefined;
                  return (
                    <tr key={`${method ?? ""}/${v.variant}`}>
                      {refMethod && <td>{method}</td>}
                      {engines && <td>{engineOf(v)}</td>}
                      <td>{v.variant}</td>
                      <td>{v.n}</td>
                      <td className="num">{fmt(v.mean)}</td>
                      <td className="num">{fmt(v.std)}</td>
                      <td className="num">{fmt(v.minimum)}</td>
                      <td className="num">{fmt(v.maximum)}</td>
                      <DifferenceCells d={d} />
                      {refMethod && <DifferenceCells d={dm} />}
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      ))}
    </>
  );
}

export function PredictionPage({ path }: { path: string }) {
  const api = useApi();
  const [given, setGiven] = useState(path);
  const summary = useAsync(() => (path ? api.prediction(path) : Promise.resolve(null)), [path]);
  return (
    <section>
      <header className="page-head">
        <h1>Prediction</h1>
        <form
          onSubmit={(e) => {
            e.preventDefault();
            window.location.hash = href({ page: "prediction", path: given });
          }}
        >
          <input
            placeholder="RUN_DIR/predictions/NAME"
            value={given}
            onChange={(e) => setGiven(e.target.value)}
            size={48}
          />{" "}
          <button type="submit">Open</button>
        </form>
        <span className="grow" />
        {path && summary.data && (
          <DeleteButton target={{ prediction: path }} after={href({ page: "prediction", path: "" })} />
        )}
      </header>
      <Show state={summary}>{(s) => (s ? <PredictionView summary={s} /> : <p className="muted">Give a prediction directory.</p>)}</Show>
    </section>
  );
}

function PredictionView({ summary: s }: { summary: PredictionSummary }) {
  const stats = ["mean", "std", ...Object.keys(Object.values(Object.values(s.final)[0] ?? {})[0] ?? {}).filter((k) => !["mean", "std"].includes(k))];
  return (
    <>
      <dl className="facts">
        <dt>Name</dt>
        <dd>{s.name}</dd>
        <dt>Run</dt>
        <dd>
          <PathText path={s.run_dir} />
        </dd>
        <dt>Times</dt>
        <dd>
          {s.times.length} ({s.times[0]} to {s.times[s.times.length - 1]}); forecast from{" "}
          {s.times[s.first_prediction_time] ?? "–"}
        </dd>
      </dl>
      <h2>Values at the last time</h2>
      {s.ensembles.map((ensemble) => (
        <div key={ensemble} className="card">
          <h3>{ensemble}</h3>
          <div className="scroll">
            <table>
              <thead>
                <tr>
                  <th>Vector</th>
                  {stats.map((k) => (
                    <th key={k}>{k}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {s.vectors.map((v) => (
                  <tr key={v}>
                    <td>{v}</td>
                    {stats.map((k) => (
                      <td key={k} className="num">
                        {fmt(s.final[ensemble]?.[v]?.[k])}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      ))}
      {Object.keys(s.scores).length > 0 && (
        <>
          <h2>Forecast scores against the truth</h2>
          {Object.entries(s.scores).map(([ensemble, byScore]) => (
            <div key={ensemble} className="card">
              <h3>{ensemble}</h3>
              <div className="scroll">
                <table>
                  <thead>
                    <tr>
                      <th>Score</th>
                      {Object.keys(Object.values(byScore)[0] ?? {}).map((v) => (
                        <th key={v}>{v}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {Object.entries(byScore).map(([score, values]) => (
                      <tr key={score}>
                        <td>{score}</td>
                        {Object.values(values).map((value, i) => (
                          <td key={i} className="num">
                            {fmt(value)}
                          </td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          ))}
        </>
      )}
    </>
  );
}
