import { createContext, useCallback, useContext, useEffect, useState, type ReactNode } from "react";

import { Api, ApiError } from "../api";
import { NO_PATHS, type Paths } from "../paths";

export const ApiContext = createContext<Api | null>(null);
export const PathsContext = createContext<Paths>(NO_PATHS);

/**
 * Who is signed: the owner, or a UI user limited by ui_access.toml. "pending"
 * until the service has said (and if it cannot): no owner's view is given on a guess.
 */
export interface Who {
  role: "owner" | "ui" | "pending";
  guest?: string;
  actions: string[];
  /** One's own web-UI folder, as the workspace shows it; files there may be deleted. */
  ui_folder?: string | null;
  /** Whether the service could not say (whoami failed): still "pending", never the owner's view. */
  failed?: boolean;
}
export const WhoContext = createContext<Who>({ role: "pending", actions: [] });

/** Whether a workspace path lies in the signed-in user's own UI folder. */
export function useInOwnFolder(path: string): boolean {
  const folder = useWho().ui_folder;
  return !!folder && path.startsWith(folder + "/");
}

/** Who is signed in: the owner, or a web-UI guest. */
export function useWho(): Who {
  return useContext(WhoContext);
}

/** Whether the signed-in token may do an action (submit, stop, resume, edit). */
export function useCan(action: string): boolean {
  return useWho().actions.includes(action);
}

/** The workspace's paths: show server paths relative to it. */
export function usePaths(): Paths {
  return useContext(PathsContext);
}

export function useApi(): Api {
  const api = useContext(ApiContext);
  if (!api) throw new Error("no API: sign in first");
  return api;
}

export interface Async<T> {
  data: T | undefined;
  error: Error | undefined;
  loading: boolean;
  reload: () => void;
}

/** Run `load` when `deps` change (and on reload), keeping the last result while loading. */
export function useAsync<T>(load: () => Promise<T>, deps: unknown[]): Async<T> {
  const [data, setData] = useState<T>();
  const [error, setError] = useState<Error>();
  const [loading, setLoading] = useState(true);
  const [tick, setTick] = useState(0);
  useEffect(() => {
    let live = true;
    setLoading(true);
    load().then(
      (value) => {
        if (!live) return;
        setData(value);
        setError(undefined);
        setLoading(false);
      },
      (err: unknown) => {
        if (!live) return;
        setError(err instanceof Error ? err : new Error(String(err)));
        setLoading(false);
      },
    );
    return () => {
      live = false;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [...deps, tick]);
  const reload = useCallback(() => setTick((t) => t + 1), []);
  return { data, error, loading, reload };
}

/** Reload every `ms` while `active`. */
export function usePolling(reload: () => void, ms: number, active: boolean): void {
  useEffect(() => {
    if (!active) return;
    const timer = window.setInterval(reload, ms);
    return () => window.clearInterval(timer);
  }, [reload, ms, active]);
}

export function ErrorBox({ error }: { error: Error | undefined }) {
  const paths = usePaths();
  if (!error) return null;
  const extra = error instanceof ApiError ? error.problem.errors : undefined;
  return (
    <div className="error" role="alert">
      <strong>{error instanceof ApiError ? error.problem.title : "Error"}:</strong> {paths.display(error.message)}
      {extra && (
        <ul>
          {extra.map((e, i) => (
            <li key={i}>
              <code>{e.location}</code>: {paths.display(e.message)}
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

export function Show<T>({ state, children }: { state: Async<T>; children: (data: T) => ReactNode }) {
  if (state.error && state.data === undefined) return <ErrorBox error={state.error} />;
  if (state.data === undefined) return <p className="muted">Loading…</p>;
  return (
    <>
      {state.error && <ErrorBox error={state.error} />}
      {children(state.data)}
    </>
  );
}

const STATE_ICON: Record<string, string> = {
  queued: "◷",
  running: "▶",
  complete: "✓",
  stopped: "■",
  failed: "✕",
  cancelled: "⊘",
};

/** A job state: an icon and its name, never colour alone. */
export function StateBadge({ state }: { state: string }) {
  return (
    <span className={`badge state-${state}`}>
      <span aria-hidden="true">{STATE_ICON[state] ?? "•"}</span> {state}
    </span>
  );
}

export function Time({ iso }: { iso: string | null | undefined }) {
  if (!iso) return <span className="muted">–</span>;
  const date = new Date(iso);
  return <time dateTime={iso} title={iso}>{Number.isNaN(date.getTime()) ? iso : date.toLocaleString()}</time>;
}

/** A server path, shown relative to the workspace when it lies in it. */
export function PathText({ path }: { path: string }) {
  const shown = usePaths().path(path);
  return <code title={shown === path ? undefined : "in the workspace"}>{shown}</code>;
}

export function Short({ id, n = 12 }: { id: string; n?: number }) {
  return (
    <code title={id} className="id">
      {id.slice(0, n)}
    </code>
  );
}
