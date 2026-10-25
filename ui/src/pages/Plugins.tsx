import { useState, type KeyboardEvent } from "react";

import type { PluginInfo } from "../api";
import { toggled } from "../comparison";
import { Show, useApi, useAsync } from "../components/common";
import {
  filterGroup,
  groupPlugins,
  optionRows,
  registryLabel,
  summaryParts,
  type WorkflowGroup,
} from "../plugins";
import { href } from "../router";
import { withoutAdr } from "../text";

/**
 * Every plugin, by the workflow it serves (a tab each), with one plugin's details at a time:
 * `#/plugins/<workflow>/<registry>/<name>`.
 */
export function PluginsPage({ workflow, registry, name }: { workflow?: string; registry?: string; name?: string }) {
  const api = useApi();
  const plugins = useAsync(() => api.plugins(), []);
  return (
    <section>
      <h1>Plugins</h1>
      <Show state={plugins}>
        {(all) => {
          const groups = groupPlugins(all);
          const group = groups.find((g) => g.workflow.key === workflow) ?? groups[0];
          if (!group) return <p className="muted">The service lists no plugins.</p>;
          const selected =
            group.registries.find((r) => r.registry === registry)?.plugins.find((p) => p.name === name) ??
            group.registries[0]?.plugins[0];
          return (
            <>
              <WorkflowTabs groups={groups} current={group.workflow.key} />
              <div
                className="plugin-browser"
                role="tabpanel"
                id={`workflow-${group.workflow.key}`}
                aria-labelledby={`workflow-tab-${group.workflow.key}`}
              >
                <PluginList key={group.workflow.key} group={group} selected={selected} />
                {selected && <PluginDetails plugin={selected} />}
              </div>
            </>
          );
        }}
      </Show>
    </section>
  );
}

/** Move the focus among a tab list's tabs with the arrow keys, and follow it (WAI-ARIA tabs). */
function arrowKeys(keys: [string, string]) {
  return (e: KeyboardEvent<HTMLElement>) => {
    const step = e.key === keys[1] ? 1 : e.key === keys[0] ? -1 : 0;
    if (!step) return;
    const tabs = [...e.currentTarget.querySelectorAll<HTMLAnchorElement>('[role="tab"]')];
    const at = tabs.indexOf(document.activeElement as HTMLAnchorElement);
    const next = tabs[(at + step + tabs.length) % tabs.length];
    if (!next) return;
    e.preventDefault();
    next.focus();
    next.click();
  };
}

function WorkflowTabs({ groups, current }: { groups: WorkflowGroup[]; current: string }) {
  return (
    <div className="tabs" role="tablist" aria-label="Workflows" onKeyDown={arrowKeys(["ArrowLeft", "ArrowRight"])}>
      {groups.map((g) => {
        const active = g.workflow.key === current;
        return (
          <a
            key={g.workflow.key}
            id={`workflow-tab-${g.workflow.key}`}
            role="tab"
            aria-selected={active}
            aria-controls={`workflow-${g.workflow.key}`}
            tabIndex={active ? 0 : -1}
            className={active ? "tab current" : "tab"}
            href={href({ page: "plugins", workflow: g.workflow.key })}
          >
            {g.workflow.label} <span className="count">{g.count}</span>
          </a>
        );
      })}
    </div>
  );
}

const COLLAPSED_KEY = "apex.plugins.collapsed";

function loadCollapsed(): Set<string> {
  // Storage may be unavailable (a private window, blocked site data): then all start expanded.
  try {
    const stored = JSON.parse(window.localStorage.getItem(COLLAPSED_KEY) ?? "[]") as unknown;
    return new Set(Array.isArray(stored) ? stored.filter((x): x is string => typeof x === "string") : []);
  } catch {
    return new Set();
  }
}

function saveCollapsed(collapsed: Set<string>): void {
  try {
    window.localStorage.setItem(COLLAPSED_KEY, JSON.stringify([...collapsed]));
  } catch {
    // Not remembered; the page works the same.
  }
}

/** A workflow's plugins, one tab each, under their registries (collapsible, searchable). */
function PluginList({ group, selected }: { group: WorkflowGroup; selected: PluginInfo | undefined }) {
  const [search, setSearch] = useState("");
  const [collapsed, setCollapsed] = useState<Set<string>>(() => loadCollapsed());
  const update = (next: Set<string>) => {
    setCollapsed(next);
    saveCollapsed(next);
  };
  const registries = filterGroup(group, search);
  const searching = search.trim() !== "";
  return (
    <aside className="plugin-list">
      <input
        type="search"
        placeholder={`Search ${group.workflow.label.toLowerCase()} plugins`}
        aria-label="Search plugins"
        value={search}
        onChange={(e) => setSearch(e.target.value)}
      />
      <div
        role="tablist"
        aria-orientation="vertical"
        aria-label={`${group.workflow.label} plugins`}
        onKeyDown={arrowKeys(["ArrowUp", "ArrowDown"])}
      >
        {registries.length === 0 && <p className="muted">No plugin matches.</p>}
        {registries.map(({ registry, plugins }) => {
          // While searching, every registry with a match is open.
          const expanded = searching || !collapsed.has(registry);
          return (
            <div key={registry} className="registry">
              <h3 className="collapsible">
                <button
                  className="toggle"
                  aria-expanded={expanded}
                  disabled={searching}
                  onClick={() => update(toggled(collapsed, registry))}
                >
                  <span className="chevron" aria-hidden="true">
                    {expanded ? "▾" : "▸"}
                  </span>{" "}
                  {registryLabel(registry)} <span className="muted">({plugins.length})</span>
                </button>
              </h3>
              {expanded &&
                plugins.map((p) => {
                  const active = p === selected;
                  return (
                    <a
                      key={p.name}
                      role="tab"
                      aria-selected={active}
                      tabIndex={active ? 0 : -1}
                      className={active ? "plugin-tab current" : "plugin-tab"}
                      href={href({ page: "plugins", workflow: group.workflow.key, registry, name: p.name })}
                    >
                      <code>{p.name}</code>
                      {p.status && <span className="badge">{p.status}</span>}
                    </a>
                  );
                })}
            </div>
          );
        })}
      </div>
    </aside>
  );
}

function PluginDetails({ plugin: p }: { plugin: PluginInfo }) {
  const rows = optionRows(p.options);
  return (
    <article className="plugin-details card" aria-live="polite">
      <header className="page-head">
        <h2>
          <code>{p.name}</code>
        </h2>
        <span className="tag">{registryLabel(p.registry)}</span>
        {p.status && <span className="badge">{p.status}</span>}
      </header>
      <p>
        {summaryParts(withoutAdr(p.summary)).map((part, i) => (part.code ? <code key={i}>{part.text}</code> : part.text))}
      </p>
      <h3>Options</h3>
      {rows.length === 0 ? (
        <p className="muted">None.</p>
      ) : (
        <div className="scroll">
          <table>
            <thead>
              <tr>
                <th>Option</th>
                <th>Type</th>
                <th>Default</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((r) => (
                <tr key={r.name}>
                  <td>
                    <code>{r.name}</code>
                    {r.description && (
                      <div className="muted">
                        {summaryParts(withoutAdr(r.description)).map((part, i) =>
                          part.code ? <code key={i}>{part.text}</code> : part.text,
                        )}
                      </div>
                    )}
                  </td>
                  <td>
                    {r.type}
                    {r.choices && (
                      <div className="muted">
                        {r.choices.map((c, i) => (
                          <span key={c}>
                            {i > 0 && " · "}
                            <code>{c}</code>
                          </span>
                        ))}
                      </div>
                    )}
                  </td>
                  <td>{r.required ? <span className="badge">required</span> : <code>{r.default ?? "–"}</code>}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
      {p.supplied.length > 0 && (
        <p className="muted">
          Given by the workflow, not by the configuration:{" "}
          {p.supplied.map((s, i) => (
            <span key={s}>
              {i > 0 && ", "}
              <code>{s}</code>
            </span>
          ))}
        </p>
      )}
      <details>
        <summary>JSON schema</summary>
        <pre className="schema">{withoutAdr(JSON.stringify(p.options, null, 2))}</pre>
      </details>
    </article>
  );
}
