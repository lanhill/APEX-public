import { useCallback, useState } from "react";

import { ApiError, type JobHandle, type SurrogateBuildSummary, type SurrogateRow } from "../api";
import { fmt } from "../chart";
import { SurrogateAgents } from "../components/Agents";
import { DeleteButton } from "../components/Delete";
import {
  ErrorBox,
  PathText,
  Short,
  Show,
  StateBadge,
  Time,
  useApi,
  useAsync,
  useCan,
  usePaths,
  usePolling,
} from "../components/common";
import { href } from "../router";
import { surrogateLabel } from "../surrogate";

const KIND: Record<SurrogateRow["kind"], string> = {
  surrogate_build: "build",
  surrogate_build_resume: "build (resumed)",
  surrogate_check: "check",
};

/** Every surrogate job the signed-in user may see, newest first. */
export function SurrogatesPage() {
  const api = useApi();
  const rows = useAsync(() => api.surrogates(), []);
  const active = (rows.data ?? []).some((r) => r.state === "queued" || r.state === "running");
  usePolling(rows.reload, 5000, active);
  return (
    <section>
      <h1>Surrogates</h1>
      <p className="muted">
        Surrogate models fitted on full simulator runs and gated on held-out ones. Submit a build from the Editor: a
        file with <code>[surrogate]</code> and <code>[design]</code> tables. Stop or resume it from its job; check a
        model on new runs from its page.
      </p>
      <Show state={rows}>{(all) => <SurrogateTable rows={all} />}</Show>
    </section>
  );
}

function SurrogateTable({ rows }: { rows: SurrogateRow[] }) {
  if (!rows.length) return <p className="muted">No surrogate build yet.</p>;
  const newest = [...rows].sort((a, b) => (b.created ?? "").localeCompare(a.created ?? ""));
  return (
    <div className="scroll">
      <table>
        <thead>
          <tr>
            <th>Build</th>
            <th>Job</th>
            <th>State</th>
            <th>Created</th>
            <th>Finished</th>
            <th>Job id</th>
          </tr>
        </thead>
        <tbody>
          {newest.map((r) => (
            <tr key={r.job}>
              <td>
                {r.path && r.kind !== "surrogate_check" ? (
                  <a href={href({ page: "surrogate", path: r.path })} title={r.path}>
                    {surrogateLabel(r.path)}
                  </a>
                ) : r.path ? (
                  <PathText path={r.path} />
                ) : (
                  "–"
                )}
              </td>
              <td>{KIND[r.kind]}</td>
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

/** One build: its runs, gate and timings, a check on new runs, and its agents. */
export function SurrogatePage({ path }: { path: string }) {
  const api = useApi();
  const summary = useAsync(() => api.surrogateSummary(path), [path]);
  const jobs = useAsync(() => api.surrogates(), []);
  // The newest build job of this folder: its page has stop (while running) and resume.
  const job = [...(jobs.data ?? [])]
    .filter((r) => r.path === path && r.kind !== "surrogate_check")
    .sort((a, b) => (b.created ?? "").localeCompare(a.created ?? ""))[0];
  const active = job?.state === "queued" || job?.state === "running";
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
        <a href={href({ page: "surrogates" })}>Back to Surrogates</a>
      </p>
      {summary.error && active ? (
        // The summary is written after the first batch of runs.
        <p className="muted">The build is running its first runs; its summary follows.</p>
      ) : (
        <Show state={summary}>{(s) => <SurrogateView summary={s} job={job} path={path} />}</Show>
      )}
      <SurrogateAgents path={path} />
    </section>
  );
}

function SurrogateView({ summary: s, job, path }: { summary: SurrogateBuildSummary; job: SurrogateRow | undefined; path: string }) {
  const paths = usePaths();
  const canSubmit = useCan("submit");
  const gate = s.gate;
  const speedup =
    s.prediction_seconds && s.simulator_median_seconds ? s.simulator_median_seconds / s.prediction_seconds : null;
  return (
    <>
      <header className="page-head">
        <h1>{s.name}</h1>
        <span className="badge">build: {s.status}</span>
        {gate && <span className={gate.passed ? "badge" : "badge error-text"}>{gate.passed ? "gate passed" : "gate failed"}</span>}
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
        {job && !["queued", "running"].includes(job.state) && (
          <DeleteButton target={{ job: job.job }} after={href({ page: "surrogates" })} />
        )}
      </header>
      <dl className="facts">
        <dt>Surrogate</dt>
        <dd>
          <code>{s.plugin}</code>, answering {s.keys} keys ({s.datatypes.join(", ")})
        </dd>
        <dt>Design</dt>
        <dd>
          <code>{s.plan}</code>: {s.training_runs} training and {s.held_out_runs} held-out runs
          {s.coarse_only_runs ? `, ${s.coarse_only_runs} on the coarse stack only` : ""}
        </dd>
        <dt>Runs</dt>
        <dd>
          {/* As of the build's last stop or end (the summary's): a resume adds to them. */}
          {s.simulator_runs} simulator runs
          {s.failed.length > 0 && <span className="error-text"> · {s.failed.length} failed</span>}
          {s.pending > 0 && ` · ${s.pending} runs pending (resume the job to run them)`}
        </dd>
        {s.fit_seconds !== null && (
          <>
            <dt>Fit</dt>
            <dd>{fmt(s.fit_seconds, 3)} s</dd>
          </>
        )}
        {speedup !== null && (
          <>
            <dt>Prediction</dt>
            <dd>
              {fmt(s.prediction_seconds, 3)} s a member by the surrogate alone, against a median full run of{" "}
              {fmt(s.simulator_median_seconds, 3)} s ({fmt(speedup, 3)} times faster)
              <div className="muted">
                A model that also simulates each member saves less: the monitor counts a coarse run; an early one is not timed.
              </div>
            </dd>
          </>
        )}
        {s.model && (
          <>
            <dt>Model</dt>
            <dd>{paths.path(`${path}/${s.model}`)}</dd>
          </>
        )}
      </dl>
      {gate && (
        <>
          <h2>Gate</h2>
          <p className="muted">On the {gate.held_out.length} held-out runs, never fitted on. Runs use the model only if it passed.</p>
          <div className="scroll">
            <table>
              <thead>
                <tr>
                  <th>Datatype</th>
                  <th>Measure</th>
                  <th>Error</th>
                  <th>Tolerance</th>
                  <th>Passed</th>
                </tr>
              </thead>
              <tbody>
                {gate.datatypes.map((d) => (
                  <tr key={d.datatype}>
                    <td>{d.datatype}</td>
                    <td>
                      <code>{d.measure}</code>
                    </td>
                    <td>{d.error === null ? "not finite" : fmt(d.error, 4)}</td>
                    <td>{fmt(d.tolerance, 4)}</td>
                    <td className={d.passed ? "" : "error-text"}>{d.passed ? "yes" : "no"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          {gate.rank && (
            <p>
              The held-out objectives' rank correlation: {gate.rank.correlation === null ? "undefined" : fmt(gate.rank.correlation, 4)}{" "}
              (at least {fmt(gate.rank.tolerance, 3)}: {gate.rank.passed ? "passed" : "failed"}).
            </p>
          )}
        </>
      )}
      {s.model && s.status === "complete" && canSubmit && <CheckForm model={`${path}/${s.model}`} />}
    </>
  );
}

/** Gate the model on new full runs of a case: a check job (history matching refuses a model whose last check failed). */
function CheckForm({ model }: { model: string }) {
  const api = useApi();
  const paths = usePaths();
  const [against, setAgainst] = useState("");
  const [runs, setRuns] = useState(10);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<Error>();
  const [token, setToken] = useState<string | null>(null);
  const [submitted, setSubmitted] = useState<JobHandle>();
  const submit = async (approval?: string) => {
    setBusy(true);
    setError(undefined);
    try {
      // A workspace-relative path, as the Editor shows them, goes as the server's.
      const where = paths.toServer(against.trim());
      setSubmitted(await api.checkSurrogate({ model_file: model, against: where, runs }, approval));
      setToken(null);
    } catch (err) {
      if (err instanceof ApiError && err.status === 403 && err.problem.token) setToken(err.problem.token);
      else setError(err as Error);
    } finally {
      setBusy(false);
    }
  };
  return (
    <section className="card">
      <h2>Check on new runs</h2>
      <p className="muted">
        Runs the case's simulator on new members (a configuration's prior, or a run folder's posterior) and gates the
        model on them, beside it in <code>checks/</code>.
      </p>
      <div className="row">
        <label>
          Case or run folder{" "}
          <input value={against} onChange={(e) => setAgainst(e.target.value)} placeholder="a .toml file or a run folder" />
        </label>{" "}
        <label>
          Runs{" "}
          <input type="number" min={1} value={runs} onChange={(e) => setRuns(Math.max(1, Number(e.target.value) || 1))} />
        </label>{" "}
        <button disabled={busy || !against} onClick={() => void submit()}>
          {busy ? "Submitting…" : "Check"}
        </button>
      </div>
      {token && (
        <p className="note">
          This check needs approval.{" "}
          <button disabled={busy} onClick={() => void submit(token)}>
            Approve and submit
          </button>
        </p>
      )}
      {submitted && (
        <p>
          Submitted:{" "}
          <a href={href({ page: "job", id: submitted.id })}>
            job <Short id={submitted.id} />
          </a>
        </p>
      )}
      <ErrorBox error={error} />
    </section>
  );
}
