import { StreamLanguage } from "@codemirror/language";
import { toml } from "@codemirror/legacy-modes/mode/toml";
import { EditorView, basicSetup } from "codemirror";
import { useCallback, useEffect, useRef, useState } from "react";
import { parse as parseToml } from "smol-toml";

import { ApiError, type FileEntry, type JobHandle, type OptimizationPlan, type Validation } from "../api";
import { fmt } from "../chart";
import { anyOtherEngine, engineOf } from "../comparison";
import { ErrorBox, Show, useApi, useAsync, useCan, useInOwnFolder, usePaths } from "../components/common";
import { href } from "../router";
import { submittedNote } from "../runs";
import { GeomodelPrior, PlanStudy } from "../components/Agents";
import { Markdown } from "../components/Chat";
import { isReadOnly, kindOf, syntaxError } from "../toml";

export function EditorPage({ path }: { path?: string }) {
  const canEdit = useCan("edit");
  return (
    <section className="editor-page">
      <aside className="tree">
        <h2>Workspace</h2>
        <Tree key={path ?? ""} selected={path} />
      </aside>
      <div className="editor-main">
        {path ? (
          isReadOnly(path) ? (
            <ReportView key={path} path={path} />
          ) : (
            <FileEditor key={path} path={path} />
          )
        ) : canEdit ? (
          <NewFile />
        ) : (
          <p className="muted">Open a file from the workspace.</p>
        )}
      </div>
    </section>
  );
}

/** A saved report, read-only and rendered (no save, validate or submit). */
function ReportView({ path }: { path: string }) {
  const api = useApi();
  const file = useAsync(() => api.readFile(path), [path]);
  return (
    <>
      <header className="page-head">
        <h1 className="path">
          <code>{path}</code>
        </h1>
        <span className="badge">read-only</span>
      </header>
      <p className="muted">A report the assistant saved. The editor writes only configurations (.toml).</p>
      <Show state={file}>
        {(f) => (
          <article className="card report">
            <Markdown text={f.text} />
          </article>
        )}
      </Show>
    </>
  );
}

function Tree({ selected }: { selected?: string }) {
  const api = useApi();
  const top = useAsync(() => api.files(), []);
  return (
    <Show state={top}>
      {(entries) => (
        <ul className="tree-list">
          <Entries items={entries} selected={selected} />
        </ul>
      )}
    </Show>
  );
}

/** Folders and files of one folder; paths are relative to the workspace. */
function Entries({ items, selected }: { items: FileEntry[]; selected?: string }) {
  if (!items.length) return <li className="muted">empty</li>;
  return (
    <>
      {items.map((c) =>
        c.kind === "folder" ? (
          <Folder key={c.path} entry={c} selected={selected} />
        ) : (
          <li key={c.path}>
            <a className={c.path === selected ? "current" : ""} href={href({ page: "editor", path: c.path })}>
              {c.name}
            </a>
          </li>
        ),
      )}
    </>
  );
}

function Folder({ entry, selected }: { entry: FileEntry; selected?: string }) {
  const api = useApi();
  const [open, setOpen] = useState(selected?.startsWith(entry.path + "/") ?? false);
  const children = useAsync(() => (open ? api.files(entry.path) : Promise.resolve([])), [open, entry.path]);
  return (
    <li>
      <button className="link folder" onClick={() => setOpen(!open)} aria-expanded={open}>
        {open ? "▾" : "▸"} {entry.name}
      </button>
      {open && (
        <Show state={children}>
          {(items) => (
            <ul className="tree-list">
              <Entries items={items} selected={selected} />
            </ul>
          )}
        </Show>
      )}
    </li>
  );
}

function NewFile() {
  const api = useApi();
  const roots = useAsync(() => api.workspace(), []);
  const [path, setPath] = useState("");
  const [error, setError] = useState<Error>();
  useEffect(() => {
    const first = roots.data?.roots[0];
    if (!path && first) setPath(first.name ? `${first.name}/new_run.toml` : "new_run.toml");
  }, [roots.data, path]);
  const create = async () => {
    try {
      await api.saveFile(path, "# A history-matching run (see examples/linear_gaussian/)\n", null);
      window.location.hash = href({ page: "editor", path });
    } catch (err) {
      setError(err as Error);
    }
  };
  return (
    <>
      <h1>Configuration editor</h1>
      <p>
        Open a <code>.toml</code> file from the workspace, or create one (a path relative to the workspace). The
        editor writes only <code>.toml</code> files under the workspace roots (<code>[workspace]</code> in{" "}
        <code>service.toml</code>).
      </p>
      <form
        className="row"
        onSubmit={(e) => {
          e.preventDefault();
          void create();
        }}
      >
        <input value={path} onChange={(e) => setPath(e.target.value)} size={60} aria-label="new file path" />
        <button type="submit" disabled={!path.endsWith(".toml")}>
          Create
        </button>
      </form>
      <ErrorBox error={error ?? roots.error} />
    </>
  );
}

function FileEditor({ path }: { path: string }) {
  const api = useApi();
  const paths = usePaths();
  const canEdit = useCan("edit");
  const canSubmit = useCan("submit");
  const mayDelete = useCan("delete");
  const inOwnFolder = useInOwnFolder(path); // hooks are called unconditionally
  const canDelete = mayDelete && inOwnFolder;
  const [confirming, setConfirming] = useState(false);
  const host = useRef<HTMLDivElement>(null);
  const view = useRef<EditorView | null>(null);
  const [sha, setSha] = useState<string | null>(null);
  const [saved, setSaved] = useState("");
  const [text, setText] = useState("");
  const [error, setError] = useState<Error>();
  const [validation, setValidation] = useState<Validation | null>(null);
  const [busy, setBusy] = useState(false);
  const [pending, setPending] = useState<string | null>(null);
  const [submitted, setSubmitted] = useState<JobHandle | null>(null);
  const [version, setVersion] = useState(0);
  const dirty = text !== saved;
  const syntax = syntaxError(text);

  useEffect(() => {
    let live = true;
    api.readFile(path).then(
      (file) => {
        if (!live || !host.current) return;
        setSha(file.sha256);
        setSaved(file.text);
        setText(file.text);
        view.current?.destroy();
        view.current = new EditorView({
          doc: file.text,
          parent: host.current,
          extensions: [
            basicSetup,
            StreamLanguage.define(toml),
            EditorView.updateListener.of((u) => {
              if (u.docChanged) setText(u.state.doc.toString());
            }),
          ],
        });
      },
      (err) => live && setError(err as Error),
    );
    return () => {
      live = false;
      view.current?.destroy();
      view.current = null;
    };
  }, [api, path, version]);

  const save = useCallback(async () => {
    setBusy(true);
    setError(undefined);
    try {
      const result = await api.saveFile(path, text, sha);
      setSha(result.sha256);
      setSaved(text);
    } catch (err) {
      setError(err as Error);
    } finally {
      setBusy(false);
    }
  }, [api, path, text, sha]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if ((e.ctrlKey || e.metaKey) && e.key === "s") {
        e.preventDefault();
        if (dirty) void save();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [dirty, save]);

  const validate = async () => {
    setBusy(true);
    setError(undefined);
    try {
      setValidation(await api.validateText(text, paths.folderOf(path)));
    } catch (err) {
      setError(err as Error);
    } finally {
      setBusy(false);
    }
  };

  const submit = async (token?: string) => {
    setBusy(true);
    setError(undefined);
    try {
      const kind = kindOf(parseToml(text) as Record<string, unknown>);
      const file = paths.toServer(path);
      const handle =
        kind === "study"
          ? await api.submitStudy(file, token)
          : kind === "prediction"
            ? await api.submitPrediction(file)
            : kind === "optimization"
              ? await api.submitOptimization(file, token)
              : await api.submitRun(file, token);
      setPending(null);
      setSubmitted(handle);
    } catch (err) {
      if (err instanceof ApiError && err.status === 403 && err.problem.token) setPending(err.problem.token);
      else setError(err as Error);
    } finally {
      setBusy(false);
    }
  };

  const remove = async (current: string) => {
    setBusy(true);
    setError(undefined);
    try {
      await api.deleteFile(path, current);
      window.location.hash = href({ page: "editor" });
    } catch (err) {
      setError(err instanceof Error ? err : new Error(String(err)));
      setConfirming(false);
    } finally {
      setBusy(false);
    }
  };
  const reload = () => {
    setError(undefined);
    setVersion((v) => v + 1);
  };

  return (
    <>
      <header className="page-head">
        <h1 className="path">
          <code>{path}</code>
          {dirty && <span className="muted"> · modified</span>}
        </h1>
        <span className="grow" />
        {canEdit && (
          <button onClick={() => void save()} disabled={busy || !dirty}>
            Save
          </button>
        )}
        <button onClick={() => void validate()} disabled={busy || syntax !== null}>
          Validate
        </button>
        {canSubmit && (
          <button
            onClick={() => void submit()}
            disabled={busy || dirty || syntax !== null}
            title={dirty ? "save first: submissions run the saved file" : undefined}
          >
            Submit
          </button>
        )}
        {canDelete && sha !== null && (
          <button disabled={busy} onClick={() => setConfirming(true)}>
            Delete…
          </button>
        )}
      </header>
      {confirming && sha !== null && (
        <div className="card confirm" role="alertdialog" aria-label="Confirm deletion">
          <p>
            <strong>Delete {path} permanently?</strong> This cannot be undone
            {dirty ? ", and your unsaved changes are lost" : ""}.
          </p>
          <button className="danger" disabled={busy} onClick={() => void remove(sha)}>
            Delete permanently
          </button>{" "}
          <button disabled={busy} onClick={() => setConfirming(false)}>
            Cancel
          </button>
        </div>
      )}
      {syntax && (
        <p className="error" role="status">
          TOML: {syntax}
        </p>
      )}
      <ErrorBox error={error} />
      {error instanceof ApiError && error.status === 409 && (
        <p>
          The file changed on the service since you opened it.{" "}
          <button className="link" onClick={reload}>
            Reload it
          </button>{" "}
          (your changes here are lost), or copy them out first.
        </p>
      )}
      <div ref={host} className="code" />
      {!dirty && isRunConfig(saved) && <PlanStudy casePath={path} optimization={isOptimization(saved)} />}
      {!dirty && isRunConfig(saved) && !isOptimization(saved) && <GeomodelPrior casePath={path} />}
      {pending && (
        <div className="approval card" role="alert">
          <h3>Approval needed</h3>
          <p>
            This submission is above the service's approval limits. A person approves it on the service's machine
            (never over HTTP):
          </p>
          <pre>apex approve {pending}</pre>
          <button onClick={() => void submit(pending)} disabled={busy}>
            Approved: submit again with the token
          </button>
        </div>
      )}
      {submitted && <SubmittedNote handle={submitted} />}
      {validation && <ValidationView validation={validation} />}
    </>
  );
}

function ValidationView({ validation: v }: { validation: Validation }) {
  const paths = usePaths();
  const plans = v.plan ? [v.plan] : v.plans;
  // The engine column only when a plan runs on another engine than APEX's.
  const engines = anyOtherEngine(plans);
  return (
    <div className="card">
      <h3>{v.valid ? "✓ Valid" : "✕ Invalid"}</h3>
      {v.errors.length > 0 && (
        <ul className="error-list">
          {v.errors.map((e, i) => (
            <li key={i}>{paths.display(e)}</li>
          ))}
        </ul>
      )}
      {plans.length > 0 && (
        <div className="scroll">
          <table>
            <thead>
              <tr>
                <th>Run</th>
                <th>Method</th>
                {engines && <th>Engine</th>}
                <th>Members</th>
                <th>State size</th>
                <th>Data</th>
                <th>Forecasts ≤</th>
                <th>Simulator runs ≤</th>
                <th>Simulator</th>
                <th>Est. hours</th>
                <th title="About this much per checkpointed iterate">Checkpoint MB</th>
                <th title="About this much memory for one ensemble forecast">Forecast MB</th>
              </tr>
            </thead>
            <tbody>
              {plans.map((p) => (
                <tr key={p.run_id}>
                  <td>
                    <code title={p.run_id}>{p.run_id.slice(0, 12)}</code>
                  </td>
                  <td>{p.method}</td>
                  {engines && <td>{engineOf(p)}</td>}
                  <td>{p.members ?? "–"}</td>
                  <td>{p.n_state}</td>
                  <td>
                    {Object.entries(p.data)
                      .map(([k, n]) => `${k} ${n}`)
                      .join(", ")}
                  </td>
                  <td>{p.max_forecasts ?? "–"}</td>
                  <td>{p.max_simulator_runs ?? "–"}</td>
                  <td>
                    {p.simulator}
                    {p.licences !== null && ` (${p.licences} licences)`}
                  </td>
                  <td>{fmt(p.estimated_hours, 2)}</td>
                  <td>{fmt(p.checkpoint_mb, 3)}</td>
                  <td>{fmt(p.forecast_memory_mb, 3)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
      {v.optimization && <OptimizationPlanView plan={v.optimization} />}
      {(v.notes ?? []).map((n, i) => (
        <p key={`note-${i}`} className="note">
          {paths.display(n)}
        </p>
      ))}
      {plans.flatMap((p) => p.warnings).map((w, i) => (
        <p key={i} className="note">
          ⚠ {paths.display(w)}
        </p>
      ))}
    </div>
  );
}

/** An optimization's plan: what it will cost, before it runs. */
function OptimizationPlanView({ plan: p }: { plan: OptimizationPlan }) {
  const paths = usePaths();
  return (
    <>
      <dl className="facts">
        <dt>Optimization</dt>
        <dd>
          {p.name} · <code>{p.optimizer}</code> on <code>{p.objective}</code>
        </dd>
        <dt>Controls</dt>
        <dd>
          {p.n_controls}, over {p.n_realizations} realization{p.n_realizations === 1 ? "" : "s"}
        </dd>
        <dt>Evaluations ≤</dt>
        <dd>{p.max_evaluations ?? "– (the optimizer cannot say)"}</dd>
        <dt>Simulator runs ≤</dt>
        <dd>
          {p.max_simulator_runs ?? "–"}
          {p.simulator && ` on ${p.simulator}`}
          {p.parallel_runs !== null && `, ${p.parallel_runs} at a time`}
        </dd>
        {p.output_dir && (
          <>
            <dt>Output</dt>
            <dd>{paths.path(p.output_dir)}</dd>
          </>
        )}
      </dl>
      {p.warnings.map((w, i) => (
        <p key={i} className="note">
          ⚠ {paths.display(w)}
        </p>
      ))}
    </>
  );
}

/** Whether a saved file is an optimization configuration. */
function isOptimization(text: string): boolean {
  try {
    return kindOf(parseToml(text) as Record<string, unknown>) === "optimization";
  } catch {
    return false;
  }
}

/** Whether a saved file is a case the planner can study: a history-matching or an optimization configuration. */
function isRunConfig(text: string): boolean {
  // A history-matching case, or an optimization (the planner plans a study of
  // optimizations from it, as it plans one of runs from a case).
  try {
    const kind = kindOf(parseToml(text) as Record<string, unknown>);
    return kind === "history_match" || kind === "optimization";
  } catch {
    return false;
  }
}

/** After a submission: its job, or that an identical run's job is elsewhere (not to be opened). */
function SubmittedNote({ handle }: { handle: JobHandle }) {
  const note = submittedNote(handle);
  if (!note.link) return <p className="note">{note.text}</p>;
  return (
    <p className="note">
      Submitted: <a href={href({ page: "job", id: note.link })}>job {note.link.slice(0, 12)}</a>
      {note.text}
    </p>
  );
}
