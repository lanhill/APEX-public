import { useEffect, useMemo } from "react";

import type { AgentInfo } from "../api";
import { agentLabel, visibleAgents, whereInUi, type UiPage } from "../agents";
import { BringCase } from "../components/Agents";
import { Show, useApi, useAsync } from "../components/common";
import { href } from "../router";
import { withoutAdr } from "../text";

const PAGE_LINK: Record<UiPage, { label: string; to: () => string }> = {
  runs: { label: "Go to Runs", to: () => href({ page: "runs" }) },
  optimizations: { label: "Go to Optimizations", to: () => href({ page: "optimizations" }) },
  studies: { label: "Go to Studies", to: () => href({ page: "studies" }) },
  editor: { label: "Go to the Editor", to: () => href({ page: "editor" }) },
};

/** APEX's runtime agents: what each does and where to use it. */
export function AgentsPage({ name }: { name?: string }) {
  const api = useApi();
  const agents = useAsync(() => api.agents(), []);
  const listed = useMemo(() => agents.data && visibleAgents(agents.data), [agents.data]);
  useEffect(() => {
    // An agent's address (#/agents/<name>) has no page of its own: it leaves no stale bookmark
    // behind. location.replace fires hashchange, so the router (and the chat's page context) follow it.
    if (name) window.location.replace(href({ page: "agents" }));
  }, [name]);
  return (
    <section>
      <h1>Agents</h1>
      <p className="muted">
        APEX's agents compute their facts in code and apply named rules; a language model, when one is
        configured and asked, judges those facts, and the rules are the fallback. None of them starts or
        stops work on its own.
      </p>
      <Show state={agents}>
        {() =>
          listed?.length ? (
            <div className="agent-grid">
              {listed.map((a) => (
                <AgentCard key={a.name} agent={a} />
              ))}
            </div>
          ) : (
            <p className="muted">No agent is available to you.</p>
          )
        }
      </Show>
      <BringCase />
    </section>
  );
}

function AgentCard({ agent: a }: { agent: AgentInfo }) {
  const where = whereInUi(a.name);
  return (
    <article className="card agent-card">
      <h2>{agentLabel(a.name)}</h2>
      <p>{withoutAdr(a.summary)}</p>
      <dl className="facts">
        {where && (
          <>
            <dt>In the web UI</dt>
            <dd>
              {where.text}{" "}
              {where.pages.map((page, i) => (
                <span key={page}>
                  {i > 0 && " · "}
                  <a href={PAGE_LINK[page].to()}>{PAGE_LINK[page].label}</a>
                </span>
              ))}
            </dd>
          </>
        )}
        <dt>Command</dt>
        <dd>
          <code>{a.command}</code>
        </dd>
        {a.tool && (
          <>
            <dt>Assistant tool</dt>
            <dd>
              <code>{a.tool}</code>
            </dd>
          </>
        )}
      </dl>
    </article>
  );
}
