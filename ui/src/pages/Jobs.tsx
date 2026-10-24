import { useEffect, useRef, useState } from "react";

import { ApiError, type ApexEvent, type JobStatus } from "../api";
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
import { DeleteButton } from "../components/Delete";
import { href } from "../router";
import { reusedRun } from "../runs";
import { follow } from "../sse";

const STATES = ["", "queued", "running", "complete", "stopped", "failed", "cancelled"];

/** Optimization jobs: their directory is an optimization's, not a catalogue run. */
const OPTIMIZATION_KINDS = ["optimization", "optimization_resume"];
/** Jobs the service resumes from their checkpoint: runs and optimizations. */
const RESUMABLE = ["run", "resume", ...OPTIMIZATION_KINDS];

export function JobsPage() {
  const api = useApi();
  const [state, setState] = useState("");
  const jobs = useAsync(() => api.jobs(), []);
  usePolling(jobs.reload, 3000, true);
  return (
    <section>
      <header className="page-head">
        <h1>Jobs</h1>
        <label>
          State{" "}
          <select value={state} onChange={(e) => setState(e.target.value)}>
            {STATES.map((s) => (
              <option key={s} value={s}>
                {s || "all"}
              </option>
            ))}
          </select>
        </label>
      </header>
      <Show state={jobs}>
        {(rows) => {
          const shown = rows.filter((r) => !state || r.state === state).reverse();
          if (!shown.length) return <p className="muted">No jobs{state && ` ${state}`}.</p>;
          return (
            <div className="scroll">
              <table>
                <thead>
                  <tr>
                    <th>Job</th>
                    <th>Kind</th>
                    <th>State</th>
                    <th>Client</th>
                    <th>Priority</th>
                    <th>Created</th>
                    <th>Finished</th>
                  </tr>
                </thead>
                <tbody>
                  {shown.map((r) => (
                    <tr key={r.id}>
                      <td>
                        <a href={href({ page: "job", id: r.id })}>
                          <Short id={r.id} />
                        </a>
                      </td>
                      <td>{r.kind}</td>
                      <td>
                        <StateBadge state={r.state} />
                      </td>
                      <td>{r.client ?? "–"}</td>
                      <td>{r.priority ?? "–"}</td>
                      <td>
                        <Time iso={r.created} />
                      </td>
                      <td>
                        <Time iso={r.finished} />
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          );
        }}
      </Show>
    </section>
  );
}

const MAX_EVENTS = 2000;

/** A job's events: the stored ones, then live over SSE until the job ends. */
function useEvents(id: string): { events: ApexEvent[]; live: boolean; error: Error | undefined } {
  const api = useApi();
  const [events, setEvents] = useState<ApexEvent[]>([]);
  const [live, setLive] = useState(false);
  const [error, setError] = useState<Error>();
  const last = useRef<number>(-1);
  useEffect(() => {
    const abort = new AbortController();
    setEvents([]);
    last.current = -1;
    const run = async () => {
      // Reconnect after a dropped connection, from the last event received.
      for (let attempt = 0; attempt < 20 && !abort.signal.aborted; attempt++) {
        try {
          setLive(true);
          await follow(
            `${api.base}/v1/jobs/${encodeURIComponent(id)}/events`,
            api.headers(),
            (message) => {
              const event = JSON.parse(message.data) as ApexEvent;
              if (event.index <= last.current) return;
              last.current = event.index;
              setEvents((old) => [...old, event].slice(-MAX_EVENTS));
            },
            abort.signal,
            last.current >= 0 ? String(last.current) : undefined,
          );
          setLive(false);
          return; // the job ended and every event came
        } catch (err) {
          if (abort.signal.aborted) return;
          setLive(false);
          setError(err instanceof Error ? err : new Error(String(err)));
          await new Promise((r) => setTimeout(r, 2000));
        }
      }
    };
    void run();
    return () => abort.abort();
  }, [api, id]);
  return { events, live, error };
}

export function JobPage({ id }: { id: string }) {
  const api = useApi();
  const paths = usePaths();
  const canStop = useCan("stop");
  const canResume = useCan("resume");
  const job = useAsync(() => api.job(id), [id]);
  const running = job.data ? ["queued", "running"].includes(job.data.state) : true;
  usePolling(job.reload, 3000, running);
  const { events, live, error } = useEvents(id);
  const [actionError, setActionError] = useState<Error>();
  const [busy, setBusy] = useState(false);

  const act = async (action: () => Promise<unknown>) => {
    setBusy(true);
    setActionError(undefined);
    try {
      const result = await action();
      if (result && typeof result === "object" && "id" in result && (result as { id: string }).id !== id) {
        window.location.hash = href({ page: "job", id: (result as { id: string }).id });
      }
      job.reload();
    } catch (err) {
      setActionError(err instanceof ApiError || err instanceof Error ? err : new Error(String(err)));
    } finally {
      setBusy(false);
    }
  };

  return (
    <section>
      <Show state={job}>
        {(status: JobStatus) => (
          <>
            <header className="page-head">
              <h1>
                {status.kind} <Short id={status.id} />
              </h1>
              <StateBadge state={status.state} />
              <span className="grow" />
              {canStop && status.state === "running" && (
                <button disabled={busy} onClick={() => act(() => api.stop(id))}>
                  Stop at next checkpoint
                </button>
              )}
              {canResume && ["stopped", "failed"].includes(status.state) && RESUMABLE.includes(status.kind) && (
                <button disabled={busy} onClick={() => act(() => api.resume(id))}>
                  Resume
                </button>
              )}
              {!["queued", "running"].includes(status.state) && (
                <DeleteButton target={{ job: id }} after={href({ page: "jobs" })} />
              )}
            </header>
            <ErrorBox error={actionError} />
            <dl className="facts">
              <dt>Created</dt>
              <dd>
                <Time iso={status.created} />
              </dd>
              <dt>Started</dt>
              <dd>
                <Time iso={status.started} />
              </dd>
              <dt>Finished</dt>
              <dd>
                <Time iso={status.finished} />
              </dd>
              {status.error && (
                <>
                  <dt>Error</dt>
                  <dd className="error-text">{paths.display(status.error)}</dd>
                </>
              )}
              {Object.entries(status.annotations).map(([k, v]) => (
                <span key={k} className="contents">
                  <dt>{k}</dt>
                  <dd>{k === "study" ? <a href={href({ page: "study", name: v })}>{v}</a> : v}</dd>
                </span>
              ))}
              <ReusedRun status={status} />
              {/* A reused run's folder is said once, by the line above. */}
              {(status.reused ? [] : status.run_dirs).map((dir) => (
                <span key={dir} className="contents">
                  <dt>Directory</dt>
                  <dd>
                    {status.kind === "prediction" ? (
                      <a href={href({ page: "prediction", path: dir })}>
                        <PathText path={dir} />
                      </a>
                    ) : OPTIMIZATION_KINDS.includes(status.kind) ? (
                      <a href={href({ page: "optimization", path: dir })}>
                        <PathText path={dir} />
                      </a>
                    ) : (
                      <PathText path={dir} />
                    )}
                  </dd>
                </span>
              ))}
            </dl>
            <RunLinks status={status} />
            {Object.keys(status.children).length > 0 && (
              <>
                <h2>Cells</h2>
                <ul className="chips">
                  {Object.entries(status.children).map(([child, state]) => (
                    <li key={child}>
                      <a href={href({ page: "job", id: child })}>
                        <Short id={child} />
                      </a>{" "}
                      <StateBadge state={state} />
                    </li>
                  ))}
                </ul>
              </>
            )}
          </>
        )}
      </Show>
      <h2>
        Events {live && <span className="live">● live</span>}
      </h2>
      <ErrorBox error={error} />
      <EventList events={events} />
    </section>
  );
}

/** The identical run a job reused: its page, or where it is. */
function ReusedRun({ status }: { status: JobStatus }) {
  const reused = reusedRun(status);
  if (!reused) return null;
  return (
    <>
      <dt>Reused</dt>
      <dd>
        {reused.text}{" "}
        {reused.link && (
          <a href={href({ page: "run", id: reused.link })}>
            <Short id={reused.link} />
          </a>
        )}
        {reused.path && <PathText path={reused.path} />}
      </dd>
    </>
  );
}

/** Links to the job's runs in the catalogue (a run's id is found by its directory). */
function RunLinks({ status }: { status: JobStatus }) {
  const api = useApi();
  const runs = useAsync(
    () =>
      status.run_dirs.length && status.kind !== "prediction" && !OPTIMIZATION_KINDS.includes(status.kind)
        ? api.runs()
        : Promise.resolve([]),
    [status.id, status.state],
  );
  const mine = (runs.data ?? []).filter((r) => status.run_dirs.includes(r.path));
  if (!mine.length) return null;
  return (
    <p>
      Results:{" "}
      {mine.map((r) => (
        <a key={r.run_id} href={href({ page: "run", id: r.run_id })}>
          {r.name} <Short id={r.run_id} n={8} />
        </a>
      ))}
    </p>
  );
}

/** An event's fields, compactly: numbers rounded, long values cut. */
function fields(values: Record<string, unknown>): string {
  return Object.entries(values)
    .filter(([k]) => !["event", "job"].includes(k))
    .map(([k, v]) => {
      const text = typeof v === "number" ? String(Number(v.toPrecision(4))) : JSON.stringify(v);
      return `${k}=${text.length > 60 ? text.slice(0, 57) + "…" : text}`;
    })
    .join(" ")
    .slice(0, 400);
}

export function EventList({ events }: { events: ApexEvent[] }) {
  const paths = usePaths();
  if (!events.length) return <p className="muted">No events yet.</p>;
  return (
    <ol className="events">
      {[...events].reverse().map((e) => (
        <li key={e.index} className={`level-${e.level.toLowerCase()}`}>
          <span className="muted">
            #{e.index} <Time iso={e.time} />
          </span>{" "}
          {e.event && <code>{e.event}</code>} {e.level !== "INFO" && <strong>{e.level}</strong>}{" "}
          {e.message.replaceAll(" ", "_") !== e.event && paths.display(e.message)}
          {Object.keys(e.fields).length > 0 && <span className="muted fields"> {paths.display(fields(e.fields))}</span>}
        </li>
      ))}
    </ol>
  );
}
