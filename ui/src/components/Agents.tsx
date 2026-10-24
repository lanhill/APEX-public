import { useEffect, useState, type ReactNode } from "react";

import type {
  AgentAnswer,
  ConvertedCase,
  CreatedCase,
  Diagnosis,
  Geomodeled,
  MonitorReport,
  PlannedStudy,
  PlotResult,
} from "../api";
import { href } from "../router";
import { Markdown } from "./Chat";
import { visibleAgents } from "../agents";
import { ErrorBox, useApi, useAsync, useInOwnFolder, useWho } from "./common";
import { withoutAdr } from "../text";

/** The agents this token may use and the web UI shows, and whether the list has loaded (or failed). */
export interface AgentList {
  names: Set<string>;
  loaded: boolean;
}

/** The agents this token may use and the web UI shows (the server's list, less private ones). */
export function useAgentList(): AgentList {
  const api = useApi();
  const agents = useAsync(() => api.agents(), []);
  return {
    names: new Set(visibleAgents(agents.data ?? []).map((a) => a.name)),
    loaded: agents.data !== undefined || agents.error !== undefined,
  };
}

function useAgentNames(): Set<string> {
  return useAgentList().names;
}

/** Who answered, for a person: the rules, or the model (and why not, if it was asked). */
export function answeredBy(answer: AgentAnswer): string {
  const who = answer.decided_by === "model" ? "the model" : "the agent's rules";
  return answer.reason ? `${who} (${withoutAdr(answer.reason)})` : who;
}

type Result =
  | { kind: "monitor"; answer: AgentAnswer & { report: MonitorReport } }
  | { kind: "diagnose"; answer: AgentAnswer & { diagnosis: Diagnosis } }
  | { kind: "report"; answer: AgentAnswer & { markdown: string } }
  | { kind: "plot"; answer: AgentAnswer & PlotResult; run: string; started: number };

function useRunner(): {
  busy: string | null;
  error: Error | undefined;
  run: <T>(label: string, call: () => Promise<T>, done: (value: T) => void) => Promise<void>;
} {
  const [busy, setBusy] = useState<string | null>(null);
  const [error, setError] = useState<Error>();
  const run = async <T,>(label: string, call: () => Promise<T>, done: (value: T) => void) => {
    setBusy(label);
    setError(undefined);
    try {
      done(await call());
    } catch (err) {
      setError(err instanceof Error ? err : new Error(String(err)));
    } finally {
      setBusy(null);
    }
  };
  return { busy, error, run };
}

/** The run's agents: monitor, diagnose, report and plot, each a button. */
export function RunAgents({ run, names }: { run: string; names: Set<string> }) {
  const api = useApi();
  const [model, setModel] = useState(false);
  const [request, setRequest] = useState("");
  const [result, setResult] = useState<Result>();
  const { busy, error, run: go } = useRunner();
  if (!["monitor", "diagnosis", "report", "plotter"].some((n) => names.has(n))) return null;
  return (
    <section className="card agents">
      <h2>Agents</h2>
      <p className="muted">They read the run and advise; none changes or stops it.</p>
      <div className="row">
        {names.has("monitor") && (
          <button disabled={busy !== null} onClick={() => void go("monitor", () => api.agentMonitor(run, model),
            (answer) => setResult({ kind: "monitor", answer }))}>
            {busy === "monitor" ? "Checking…" : "Check health"}
          </button>
        )}{" "}
        {names.has("diagnosis") && (
          <button disabled={busy !== null} onClick={() => void go("diagnose", () => api.agentDiagnose(run, model),
            (answer) => setResult({ kind: "diagnose", answer }))}>
            {busy === "diagnose" ? "Diagnosing…" : "Diagnose"}
          </button>
        )}{" "}
        {names.has("report") && (
          <button disabled={busy !== null} onClick={() => void go("report", () => api.agentReport({ run }, model),
            (answer) => setResult({ kind: "report", answer }))}>
            {busy === "report" ? "Writing…" : "Write report"}
          </button>
        )}{" "}
        <label>
          <input type="checkbox" checked={model} onChange={(e) => setModel(e.target.checked)} /> ask the model
          (slower)
        </label>
      </div>
      {names.has("plotter") && (
        <div className="row">
          <input
            aria-label="Figures to draw"
            placeholder="Figures to draw, e.g. rmse and layer maps (empty: the classic set)"
            value={request}
            onChange={(e) => setRequest(e.target.value)}
            size={60}
          />{" "}
          <button disabled={busy !== null} onClick={() => void go("plot", () => api.agentPlot(run, request, model),
            (answer) => setResult({ kind: "plot", answer, run, started: Date.now() }))}>
            {busy === "plot" ? "Drawing…" : "Draw figures"}
          </button>
        </div>
      )}
      <ErrorBox error={error} />
      {result && <AgentResult result={result} />}
    </section>
  );
}

/** An optimization's agents: the monitor, the diagnosis and the report, each a button. */
export function OptimizationAgents({ path }: { path: string }) {
  const api = useApi();
  const names = useAgentNames();
  const [model, setModel] = useState(false);
  const [result, setResult] = useState<Result>();
  const { busy, error, run: go } = useRunner();
  if (!names.has("monitor") && !names.has("diagnosis") && !names.has("report")) return null;
  return (
    <section className="card agents">
      <h2>Agents</h2>
      <p className="muted">They read the optimization and advise; none changes or stops it.</p>
      <div className="row">
        {names.has("monitor") && (
          <button disabled={busy !== null} onClick={() => void go("monitor", () => api.agentMonitorOptimization(path, model),
            (answer) => setResult({ kind: "monitor", answer }))}>
            {busy === "monitor" ? "Checking…" : "Check health"}
          </button>
        )}{" "}
        {names.has("diagnosis") && (
          <button disabled={busy !== null} onClick={() => void go("diagnose", () => api.agentDiagnoseOptimization(path, model),
            (answer) => setResult({ kind: "diagnose", answer }))}>
            {busy === "diagnose" ? "Diagnosing…" : "Diagnose"}
          </button>
        )}{" "}
        {names.has("report") && (
          <button disabled={busy !== null} onClick={() => void go("report", () => api.agentReport({ optimization: path }, model),
            (answer) => setResult({ kind: "report", answer }))}>
            {busy === "report" ? "Writing…" : "Write report"}
          </button>
        )}{" "}
        <label>
          <input type="checkbox" checked={model} onChange={(e) => setModel(e.target.checked)} /> ask the model
          (slower)
        </label>
      </div>
      <ErrorBox error={error} />
      {result && <AgentResult result={result} />}
    </section>
  );
}

function AgentResult({ result }: { result: Result }) {
  let body: ReactNode;
  if (result.kind === "monitor") {
    const r = result.answer.report;
    body = (
      <>
        <p>
          <strong>{r.status}</strong> · recommendation: <strong>{r.recommendation}</strong>
        </p>
        <p>{withoutAdr(r.summary)}</p>
        <Items items={r.findings.map((f) => `${f.severity}: ${f.code}: ${f.evidence}`)} />
      </>
    );
  } else if (result.kind === "diagnose") {
    const d = result.answer.diagnosis;
    body = (
      <>
        <p>
          <strong>{d.verdict}</strong>
        </p>
        <p>{withoutAdr(d.summary)}</p>
        <Items items={d.causes.map((c) => `${c.severity}: ${c.code}: ${c.evidence}`)} />
        {d.recommendations.length > 0 && <h3>Recommended changes</h3>}
        <Items items={d.recommendations.map((r) => `${r.change} (${r.rationale})`)} />
      </>
    );
  } else if (result.kind === "plot") {
    const p = result.answer;
    body = (
      <>
        <p>
          Drawing <strong>{p.preset ?? `${p.requests} figure requests`}</strong> in the background into report/plots;
          reload the page to see the figures under Figures as they are drawn.
        </p>
        <Items items={p.figures.map((f) => `${f.figure} (${f.subdir})`)} />
        <PlotProgress key={result.started} run={result.run} />
      </>
    );
  } else {
    body = <Markdown text={result.answer.markdown} />;
  }
  return (
    <div className="agent-result">
      {body}
      <p className="muted">Answered by {answeredBy(result.answer)}.</p>
    </div>
  );
}

/** How the background drawing stands, checked every few seconds until it ends. */
function PlotProgress({ run }: { run: string }) {
  const api = useApi();
  const [state, setState] = useState<{ state: string; written?: number; error?: string }>();
  useEffect(() => {
    let stopped = false;
    let failures = 0;
    const check = async () => {
      try {
        const found = await api.plotState(run);
        if (stopped) return;
        failures = 0;
        setState(found);
        if (found.state === "drawing") window.setTimeout(() => void check(), 3000);
      } catch {
        // A few retries, slower each time, then say the state is unknown.
        failures += 1;
        if (stopped) return;
        if (failures <= 5) window.setTimeout(() => void check(), 3000 * 2 ** failures);
        else setState({ state: "unknown" });
      }
    };
    void check();
    return () => {
      stopped = true;
    };
  }, [api, run]);
  if (state?.state === "unknown")
    return <p className="muted">The drawing's state is unknown; reload the page later to see the figures.</p>;
  if (!state || state.state === "drawing") return <p className="muted">Drawing…</p>;
  if (state.state === "failed") return <p className="error">The drawing failed: {state.error}</p>;
  return <p className="muted">Done: {state.written ?? 0} figures; reload the page to see them.</p>;
}

function Items({ items }: { items: string[] }) {
  if (!items.length) return null;
  return (
    <ul>
      {items.map((item) => (
        <li key={item}>{withoutAdr(item)}</li>
      ))}
    </ul>
  );
}

/** A study's narrative report (the report agent). */
export function StudyReport({ study }: { study: string }) {
  const api = useApi();
  const names = useAgentNames();
  const [model, setModel] = useState(false);
  const [answer, setAnswer] = useState<AgentAnswer & { markdown: string }>();
  const { busy, error, run } = useRunner();
  if (!names.has("report")) return null;
  return (
    <section className="card agents">
      <h2>Report</h2>
      <button disabled={busy !== null} onClick={() => void run("report", () => api.agentReport({ study }, model), setAnswer)}>
        {busy ? "Writing…" : "Write the study's report"}
      </button>{" "}
      <label>
        <input type="checkbox" checked={model} onChange={(e) => setModel(e.target.checked)} /> ask the model (slower)
      </label>
      <ErrorBox error={error} />
      {answer && (
        <div className="agent-result">
          <Markdown text={answer.markdown} />
          <p className="muted">Answered by {answeredBy(answer)}.</p>
        </div>
      )}
    </section>
  );
}

/** Plan a study for the case open in the editor: a draft in one's UI folder (the planner). */
export function PlanStudy({ casePath, optimization = false }: { casePath: string; optimization?: boolean }) {
  const api = useApi();
  const names = useAgentNames();
  const [goal, setGoal] = useState("");
  const [budget, setBudget] = useState("");
  const [model, setModel] = useState(false);
  const [plan, setPlan] = useState<AgentAnswer & PlannedStudy>();
  const { busy, error, run } = useRunner();
  if (!names.has("planner")) return null;
  const submit = () =>
    void run(
      "plan",
      () => api.agentPlan({ case: casePath, goal, budget: budget ? Number(budget) : undefined, model }),
      setPlan,
    );
  return (
    <section className="card agents">
      <h2>Plan a study of this case</h2>
      <form
        onSubmit={(e) => {
          e.preventDefault();
          submit();
        }}
      >
        <input
          placeholder={
            optimization
              ? "the goal, e.g. which optimizer reaches the best NPV within the budget?"
              : "the goal, e.g. does adaptive localization improve the RMSE?"
          }
          value={goal}
          onChange={(e) => setGoal(e.target.value)}
          size={56}
        />{" "}
        <input
          placeholder="budget (simulator runs)"
          value={budget}
          onChange={(e) => setBudget(e.target.value.replace(/[^0-9]/g, ""))}
          size={22}
        />{" "}
        <label>
          <input type="checkbox" checked={model} onChange={(e) => setModel(e.target.checked)} /> ask the model
        </label>{" "}
        <button type="submit" disabled={busy !== null || goal.trim().length < 3}>
          {busy ? "Planning…" : "Plan"}
        </button>
      </form>
      <ErrorBox error={error} />
      {plan && (
        <div className="agent-result">
          <p>
            <strong>{plan.study}</strong>: {plan.variants.length} variant(s) × {plan.replicates} replicate(s), at most{" "}
            {plan.cost ?? "?"} simulator runs (budget {plan.budget}){plan.within_budget ? "" : ": over the budget"}.
          </p>
          <p>{withoutAdr(plan.hypothesis)}</p>
          <ul>
            {plan.variants.map((v) => (
              <li key={v.name}>
                <strong>{v.name}</strong>: {withoutAdr(v.rationale)}
              </li>
            ))}
          </ul>
          <p>
            Draft: <a href={href({ page: "editor", path: plan.draft })}>{plan.draft}</a>: review it there, then
            Submit. Answered by {answeredBy(plan)}.
          </p>
        </div>
      )}
    </section>
  );
}

/** Bring a case into APEX (the case-converter): import a deck, or create a synthetic case. */
export function BringCase() {
  const api = useApi();
  const names = useAgentNames();
  // Only the owner runs a created case's truth (a simulation); guests submit it later.
  const owner = useWho().role === "owner";
  const [source, setSource] = useState("");
  const [history, setHistory] = useState("");
  const [request, setRequest] = useState("");
  const [name, setName] = useState("");
  const [observe, setObserve] = useState(false);
  const [model, setModel] = useState(false);
  const [converted, setConverted] = useState<AgentAnswer & ConvertedCase>();
  const [created, setCreated] = useState<AgentAnswer & CreatedCase>();
  const { busy, error, run } = useRunner();
  if (!names.has("case-converter")) return null;
  const named = name.trim() || undefined;
  const importDeck = () =>
    void run(
      "convert",
      () =>
        api.agentConvert({
          source: source.trim(),
          history: history.trim() || undefined,
          request: request.trim() || undefined,
          name: named,
          model,
        }),
      (r) => {
        setCreated(undefined);
        setConverted(r);
      },
    );
  const create = () =>
    void run(
      "create",
      () => api.agentCreateCase({ request: request.trim() || undefined, name: named, observe: owner && observe, model }),
      (r) => {
        setConverted(undefined);
        setCreated(r);
      },
    );
  const answer = converted ?? created;
  return (
    <section className="card agents">
      <h2>Bring a case</h2>
      <p className="muted">
        Import a deck (OPM Flow, Eclipse) with its history as a history-matching case, or create a synthetic
        case from a description. Either is a draft in your folder: nothing runs until you submit it.
      </p>
      <form
        onSubmit={(e) => {
          e.preventDefault();
          importDeck();
        }}
      >
        <input
          placeholder="the deck, e.g. Norne/NORNE.DATA"
          value={source}
          onChange={(e) => setSource(e.target.value)}
          size={40}
        />{" "}
        <input
          placeholder="its history (optional: a CSV or summary file)"
          value={history}
          onChange={(e) => setHistory(e.target.value)}
          size={36}
        />{" "}
        <button type="submit" disabled={busy !== null || source.trim().length === 0}>
          {busy === "convert" ? "Importing…" : "Import the deck"}
        </button>
      </form>
      <form
        onSubmit={(e) => {
          e.preventDefault();
          create();
        }}
      >
        <input
          placeholder="or describe a synthetic case, e.g. a 30 x 30 x 4 five-spot, 200 mD"
          value={request}
          onChange={(e) => setRequest(e.target.value)}
          size={56}
        />{" "}
        {owner && (
          <label>
            <input type="checkbox" checked={observe} onChange={(e) => setObserve(e.target.checked)} /> run its truth
            once now (one simulation)
          </label>
        )}{" "}
        <button type="submit" disabled={busy !== null}>
          {busy === "create" ? "Creating…" : "Create"}
        </button>
      </form>
      <p>
        <input placeholder="name (optional)" value={name} onChange={(e) => setName(e.target.value)} size={20} />{" "}
        <label>
          <input type="checkbox" checked={model} onChange={(e) => setModel(e.target.checked)} /> ask the model
        </label>
      </p>
      <ErrorBox error={error} />
      {answer && (
        <div className="agent-result">
          {converted && (
            <p>
              State:{" "}
              {converted.choices.variables.map((v) => `${v.property}${v.transform === "log" ? " (log)" : ""}`).join(", ")};{" "}
              {converted.choices.observations.length} observed vectors; {converted.choices.method}.
            </p>
          )}
          <p>
            {answer.valid ? "Valid." : "Not valid yet."} Draft:{" "}
            <a href={href({ page: "editor", path: answer.draft })}>{answer.draft}</a>. Answered by {answeredBy(answer)}.
          </p>
          <Items items={answer.valid ? answer.notes : [...answer.errors, ...answer.notes]} />
        </div>
      )}
    </section>
  );
}

/** Give the configuration open in the editor a prior (the geomodeler): a new configuration beside it. */
export function GeomodelPrior({ casePath }: { casePath: string }) {
  const api = useApi();
  const names = useAgentNames();
  // The new configuration is written beside this one: only in one's own folder.
  const own = useInOwnFolder(casePath);
  const [request, setRequest] = useState("");
  const [model, setModel] = useState(false);
  const [result, setResult] = useState<AgentAnswer & Geomodeled>();
  const { busy, error, run } = useRunner();
  if (!names.has("geomodeler") || !own) return null;
  const submit = () =>
    void run(
      "geomodel",
      () => api.agentGeomodel({ config: casePath, request: request.trim() || undefined, model }),
      setResult,
    );
  return (
    <section className="card agents">
      <h2>Give it a prior</h2>
      <form
        onSubmit={(e) => {
          e.preventDefault();
          submit();
        }}
      >
        <input
          placeholder="optional, e.g. ln PERMX around 5 with a 10-cell range"
          value={request}
          onChange={(e) => setRequest(e.target.value)}
          size={56}
        />{" "}
        <label>
          <input type="checkbox" checked={model} onChange={(e) => setModel(e.target.checked)} /> ask the model
        </label>{" "}
        <button type="submit" disabled={busy !== null}>
          {busy ? "Drawing…" : "Give it a prior"}
        </button>
      </form>
      <ErrorBox error={error} />
      {result && (
        <div className="agent-result">
          <p>
            {result.valid && result.qc.length === 0 ? "Valid, and a drawn sample matches." : "Not valid yet."} New
            configuration:{" "}
            <a href={href({ page: "editor", path: result.config })}>{result.config}</a> ({result.skill}). Answered by{" "}
            {answeredBy(result)}.
          </p>
          <Items items={[...result.errors, ...result.qc, ...result.notes]} />
        </div>
      )}
    </section>
  );
}
