// The catalogue of examples and toy problems, by workflow: copy one into your own folder and
// open it in the Editor, where it is validated and submitted as any configuration of yours.

import { useState } from "react";

import type { Example } from "../api";
import { ErrorBox, Show, useApi, useAsync, useCan } from "../components/common";
import { examplesByWorkflow } from "../front";
import { href } from "../router";

const COST: Record<Example["cost"], string> = {
  seconds: "runs in seconds",
  minutes: "runs in minutes",
  hours: "runs in hours",
};

export function ExamplesPage({ workflow }: { workflow?: string }) {
  const api = useApi();
  const state = useAsync(() => api.examples(), [api]);
  return (
    <>
      <header className="page-head">
        <h1>Examples</h1>
      </header>
      <p className="muted">
        Toy problems and example cases, by workflow. Copy one into your own folder: it opens in the Editor, where you
        can change it, validate it and submit it.
      </p>
      <Show state={state}>
        {(data) => {
          const tabs = examplesByWorkflow(data.examples);
          if (tabs.length === 0) return <p className="muted">This service offers no examples.</p>;
          const current = tabs.find((t) => t.key === workflow) ?? tabs[0]!;
          return (
            <>
              <div className="tabs" role="tablist" aria-label="Workflows">
                {tabs.map((t) => (
                  <a
                    key={t.key}
                    role="tab"
                    aria-selected={t.key === current.key}
                    href={href({ page: "examples", workflow: t.key })}
                    className={t.key === current.key ? "tab current" : "tab"}
                  >
                    {t.label} <span className="count">{t.examples.length}</span>
                  </a>
                ))}
              </div>
              <div className="example-grid">
                {current.examples.map((e) => (
                  <ExampleCard key={e.id} example={e} workflow={current.key} />
                ))}
              </div>
            </>
          );
        }}
      </Show>
    </>
  );
}

function ExampleCard({ example: e, workflow }: { example: Example; workflow: string }) {
  const api = useApi();
  const canCopy = useCan("examples");
  const [error, setError] = useState<Error | undefined>(undefined);
  const [busy, setBusy] = useState(false);
  const entries = e.entries.filter((x) => x.workflow === workflow);
  const copy = async (config: string) => {
    setBusy(true);
    setError(undefined);
    try {
      const copied = await api.copyExample(e.id);
      const target = copied.configs.find((c) => c.path.endsWith(`/${config}`)) ?? copied.configs[0];
      if (target) window.location.hash = href({ page: "editor", path: target.path });
    } catch (err) {
      setError(err instanceof Error ? err : new Error(String(err)));
    } finally {
      setBusy(false);
    }
  };
  return (
    <article className="card example-card">
      <h2>{e.title}</h2>
      <p>{e.summary}</p>
      <p className="muted small">
        {COST[e.cost]}
        {e.requires.length > 0 && <> · needs {e.requires.join(", ")}</>}
        {e.tags.length > 0 && <> · {e.tags.join(", ")}</>}
      </p>
      {e.source && <p className="muted small">Source: {e.source}</p>}
      <div className="actions">
        {entries.map((x) => (
          <button key={x.config} type="button" disabled={busy || !canCopy} onClick={() => void copy(x.config)}>
            Copy {x.title ? `(${x.title})` : ""} to my folder
          </button>
        ))}
      </div>
      {!canCopy && <p className="muted small">Copying examples is not enabled for you.</p>}
      <ErrorBox error={error} />
    </article>
  );
}
