// The REST API of APEX (docs/api.md), with the documents it returns. The UI
// holds no logic of its own beyond display: every number comes from here.

import type { ContributedPage } from "./contributed";

export interface Problem {
  type: string;
  title: string;
  status: number;
  detail: string;
  token?: string;
  sha256?: string | null;
  errors?: { location: string; message: string }[];
}

export class ApiError extends Error {
  constructor(
    readonly status: number,
    readonly problem: Problem,
  ) {
    super(problem.detail || problem.title);
  }
}

export type JobState = "queued" | "running" | "complete" | "stopped" | "failed" | "cancelled";

export interface JobRow {
  id: string;
  kind: string;
  state: JobState;
  created: string;
  finished: string | null;
  client: string | null;
  priority: string | null;
}

export interface JobStatus {
  id: string;
  kind: string;
  state: JobState;
  created: string;
  started: string | null;
  finished: string | null;
  error: string | null;
  run_dirs: string[];
  /** The runs the job ran or used; absent from older services. */
  run_ids?: string[];
  children: Record<string, JobState>;
  annotations: Record<string, string>;
  /** An identical run the job reused instead of running (its directory); null for a guest when it is not theirs. */
  reused?: string | null;
  /** Whether this user may open the reused run's page; null when nothing was reused. */
  reused_visible?: boolean | null;
}

export interface JobHandle {
  id: string;
  kind: string;
  run_ids: string[];
  reused: boolean;
  /** Whether this web-UI user may open the job's page (false: another submission's job); absent: not said. */
  job_visible?: boolean | null;
  job_state?: JobState | null;
}

export interface ApexEvent {
  index: number;
  time: string;
  event: string | null;
  level: string;
  message: string;
  fields: Record<string, unknown>;
}

export interface RunRow {
  run_id: string;
  name: string;
  created: string;
  path: string;
  annotations: Record<string, string>;
}

export interface RunSummary {
  run_id: string | null;
  path: string;
  name: string;
  method: string;
  state: "complete" | "stopped" | "not started";
  converged: boolean | null;
  iterations: number[];
  /** The accepted iterates' records, in order: step n is the n-th (0: the prior). */
  accepted_iterations?: number[];
  posterior_iteration: number | null;
  selected_iteration: number | null;
  members: number | null;
  data_mismatch: Record<string, [number, number]>;
  metrics: Record<string, Record<string, Record<string, number>>>;
  posterior_metrics: Record<string, Record<string, number>>;
  simulator_runs: number;
  annotations: Record<string, string>;
  provenance: Record<string, unknown> | null;
}

export interface StepInfo {
  /** The history record (iter_<k>): counts every trial, rejected ones too. */
  iteration: number;
  /** 0, 1, 2, ...: the prior, then each accepted iterate. */
  step?: number;
  n_members: number;
  rejected_before: number;
  control: number | null;
  label: string | null;
}

export interface StepStatistics {
  metric: string;
  label: string;
  iteration: number;
  mean: number | null;
  std: number | null;
  median: number | null;
  p10: number | null;
  p90: number | null;
}

export interface RunSteps {
  workflow: string;
  name: string;
  path: string;
  method: string;
  status: string;
  steps: StepInfo[];
  metrics: StepStatistics[];
  selected_iteration: number | null;
  selected_step?: number | null;
  forecasts: number;
  notes: string[];
}

export interface RunPlan {
  run_id: string;
  method: string;
  /** Where it runs: "apex", or another engine such as "ert"; absent from older services. */
  engine?: string;
  members: number | null;
  n_state: number;
  data: Record<string, number>;
  max_forecasts: number | null;
  max_simulator_runs: number | null;
  simulator: string;
  licences: number | null;
  parallel_runs: number | null;
  estimated_hours: number | null;
  /** About this much per checkpointed iterate (MB); absent from older services. */
  checkpoint_mb?: number | null;
  /** About this much memory for one ensemble forecast (MB); absent from older services. */
  forecast_memory_mb?: number | null;
  warnings: string[];
}

/** What an optimization will cost, before it runs. */
export interface OptimizationPlan {
  optimization_id: string;
  name: string;
  optimizer: string;
  objective: string;
  n_controls: number;
  n_realizations: number;
  /** Upper bound on the objective's evaluations; null: the optimizer cannot say. */
  max_evaluations: number | null;
  /** One per evaluation for a simulated objective; 0 for an analytic one. */
  max_simulator_runs: number | null;
  simulator: string | null;
  parallel_runs: number | null;
  output_dir: string | null;
  warnings: string[];
}

/** One optimization job (GET /v1/optimizations). */
export interface OptimizationRow {
  job: string;
  kind: "optimization" | "optimization_resume";
  state: JobState;
  path: string | null;
  created: string | null;
  finished: string | null;
  client: string | null;
  annotations: Record<string, string>;
}

/** One row of an optimization's history.jsonl: a start, an accepted step or an EPF update. */
export interface OptimizationStep {
  kind: string;
  iteration: number;
  x?: number[];
  objective?: number | null;
  [key: string]: unknown;
}

/** An optimization run's progress and result. */
export interface OptimizationSummary {
  path: string;
  name: string;
  state: "new" | "running" | "stopped" | "finished";
  optimizer: string;
  objective: string;
  controls: string[];
  history: OptimizationStep[];
  /** result.json, once finished. */
  result: ({ x?: number[]; names?: string[]; objective?: number; message?: string } & Record<string, unknown>) | null;
  /** Simulated evaluations (0: an analytic objective). */
  evaluations: number;
  failed_evaluations: number;
  /** Whether the objective is computed from simulations; absent from older services. */
  simulated?: boolean;
  /**
   * The objective's obj_scaling option (null: none), which divides the quantity; every
   * optimizer minimises, so a negative one maximises it. Absent from older services.
   */
  objective_scaling?: number | null;
  /** Whether EPF is configured, so values include its penalty from the start; absent from older services. */
  penalised?: boolean;
  /** Several objectives (nsga2): the front at the last checkpoint; absent otherwise and from older services. */
  pareto?: ParetoFront | null;
  /** Whether the run has several objectives, known from the start; absent from older services. */
  multi_objective?: boolean;
  /** Several objectives: their names (null: one, or not known); absent from older services. */
  objective_names?: string[] | null;
  /** [[objectives]]: each one's obj_scaling (null: none); absent from older services. */
  objective_scalings?: (number | null)[] | null;
}

/** A multi-objective run's front (pareto.json): members' controls, objectives and violation. */
export interface ParetoFront {
  objective_names: string[];
  /** Each member's controls (physical), in the run's control order. */
  x: number[][];
  /** Each member's objectives: their means over the realizations, all minimised. */
  objectives: number[][];
  violation: number[];
  /** The member shown as the run's single answer: closest to the ideal point in normalised objectives. */
  representative: number;
  hypervolume: number;
  reference_point: number[] | null;
}

/** One configuration of an example, for one workflow. */
export interface ExampleEntry {
  workflow: string;
  config: string;
  title?: string | null;
}

/** An example or toy problem of the catalogue (GET /v1/examples). */
export interface Example {
  id: string;
  title: string;
  summary: string;
  /** Where the problem or case comes from (a project, a paper). */
  source?: string | null;
  tags: string[];
  cost: "seconds" | "minutes" | "hours";
  requires: string[];
  entries: ExampleEntry[];
}

/** Where a copied example landed in one's own folder. */
export interface CopiedExample {
  folder: string;
  configs: { workflow: string; title: string | null; path: string }[];
}

export interface Validation {
  valid: boolean;
  errors: string[];
  plan: RunPlan | null;
  /** The plan of an optimization configuration; absent from older services. */
  optimization?: OptimizationPlan | null;
  plans: RunPlan[];
  /** What a study will not run, and why, e.g. "not run: <label>: <reason>". */
  notes?: string[];
}

export interface PluginInfo {
  registry: string;
  name: string;
  /** The workflow it serves ("history_matching", ...; null: none); absent from older services. */
  workflow?: string | null;
  status: string | null;
  summary: string;
  options: Record<string, unknown>;
  supplied: string[];
}

export interface VariantStats {
  /** The study's method entry; null without methods. */
  method?: string | null;
  /** Where the method ran: "apex" or e.g. "ert". */
  engine?: string;
  variant: string;
  n: number;
  mean: number;
  std: number | null;
  minimum: number;
  maximum: number;
  values: number[];
}

export interface PairedDifference {
  method?: string | null;
  variant: string;
  reference_method?: string | null;
  reference: string;
  n: number;
  mean_difference: number;
  standard_error: number | null;
  p_value: number | null;
  note: string;
}

/** What to delete: a job, or the submission of a run or study. */
export type DeletionTarget = { job: string } | { run: string } | { study: string } | { prediction: string };

export interface DeletionPlan {
  job: string;
  kind: string;
  owner: string;
  jobs: string[];
  runs: string[];
  /** Folders, as the workspace shows them. */
  dirs: string[];
  studies: string[];
  /** Runs kept because another submission uses them. */
  kept: string[];
  /** Optimization directories that go; absent from older services. */
  optimizations?: string[];
  size: number;
  /** Why it cannot be deleted now; empty when it can. */
  blocked: string[];
  token: string;
}

export interface Comparison {
  study: string;
  replicates: number;
  reference: string;
  /** The method differences are taken against (the first); null without methods. */
  reference_method?: string | null;
  metrics: { case: string; metric: string; variants: VariantStats[]; differences: PairedDifference[] }[];
  notes: string[];
  missing: string[];
  /** Cells the study does not run, and why; absent from older services. */
  skipped?: { label: string; reason: string }[];
}

export interface PredictionSummary {
  path: string;
  name: string;
  run_dir: string;
  ensembles: string[];
  vectors: string[];
  times: string[];
  first_prediction_time: number;
  final: Record<string, Record<string, Record<string, number>>>;
  scores: Record<string, Record<string, Record<string, number>>>;
}

export interface FileEntry {
  path: string;
  name: string;
  kind: "folder" | "file";
  size: number | null;
}

export interface FileContent {
  path: string;
  text: string;
  sha256: string;
}

/** What every agent route answers besides its result. */
export interface AgentAnswer {
  decided_by: string;
  reason: string | null;
}

/** What the plotter started drawing in the background. */
export interface PlotResult {
  state: string;
  requests: number;
  figures: { figure: string; subdir: string }[];
  preset: string | null;
}

export interface MonitorReport {
  status: "healthy" | "warning" | "critical";
  recommendation: string;
  findings: { code: string; severity: string; evidence: string }[];
  summary: string;
}

export interface Diagnosis {
  verdict: "healthy" | "problems" | "failed";
  causes: { code: string; severity: string; evidence: string }[];
  recommendations: { change: string; rationale: string; addresses: string[]; patch: unknown }[];
  summary: string;
}

export interface PlannedStudy {
  draft: string;
  study: string;
  hypothesis: string;
  variants: { name: string; patch: unknown; rationale: string }[];
  replicates: number;
  cost: number | null;
  budget: number;
  within_budget: boolean;
  revisions: string[];
}

/** POST /v1/agents/convert: a deck imported as an APEX case, a draft in one's UI folder. */
export interface ConvertedCase {
  draft: string;
  skill: string;
  valid: boolean;
  errors: string[];
  notes: string[];
  choices: { variables: { property: string; transform: string }[]; observations: unknown[]; method: string };
}

/** POST /v1/agents/convert/create: a synthetic case (its observations need its truth run). */
export interface CreatedCase {
  draft: string;
  valid: boolean;
  errors: string[];
  notes: string[];
}

/** POST /v1/agents/geomodel: a configuration with a new prior, beside the old one. */
export interface Geomodeled {
  config: string;
  skill: string;
  valid: boolean;
  errors: string[];
  qc: string[];
  notes: string[];
}

/** GET /v1/chats/models: the models this user's assistant may answer with. */
export interface ChatModels {
  /** `local`: the model runs on the service's machine, so a chat's text stays there. */
  models: { name: string; label: string; local?: boolean }[];
  default: string | null;
  /** The owner picks one per message; a guest's are tried in order, `note` saying so. */
  choosable: boolean;
  note: string | null;
}

export interface AgentInfo {
  name: string;
  summary: string;
  command: string;
  tool: string | null;
  adr: string;
  /** "private" agents are never shown in the web UI; absent from older services. */
  visibility?: "public" | "private";
}

export interface Chat {
  id: string;
  title: string;
  created: string;
  updated: string;
  context: Record<string, string>;
  busy: boolean;
  closed: string | null;
  turns: number;
}

export interface ChatMessage {
  n: number;
  role: "user" | "assistant";
  content: string;
  data: Record<string, unknown>;
  created: string;
}

export interface Proposal {
  n: number;
  kind: string;
  draft: string;
  summary: string;
  data: { cost?: number | null } & Record<string, unknown>;
  state: "pending" | "launched" | "discarded" | "replaced";
  job: string | null;
  created: string;
  updated: string;
}

export interface ChatEvent {
  n: number;
  created: string;
  kind: string;
  data: Record<string, unknown>;
}

export interface Page<T> {
  items: T[];
  next_cursor: string | null;
}

export type Fetch = (input: string, init?: RequestInit) => Promise<Response>;

/** A client of the REST API, sending the bearer token with every request. */
export class Api {
  constructor(
    readonly token: string,
    readonly base = "",
    private readonly fetcher: Fetch = (input, init) => fetch(input, init),
    /** Told of every 401: the token expired or is no longer accepted (the app signs out). */
    private readonly onRefused?: (error: ApiError) => void,
  ) {}

  /** The error of a failed response, after telling `onRefused` of a 401. */
  private async failure(response: Response): Promise<ApiError> {
    const error = new ApiError(response.status, await problemOf(response));
    if (response.status === 401) this.onRefused?.(error);
    return error;
  }

  headers(extra: Record<string, string> = {}): Record<string, string> {
    // The web UI names itself: with the owner's token, the server then shows only the
    // owner's own work, not the guests'.
    return { Authorization: `Bearer ${this.token}`, "X-APEX-Client": "ui", ...extra };
  }

  async request<T>(method: string, path: string, body?: unknown): Promise<T> {
    const init: RequestInit = { method, headers: this.headers() };
    if (body !== undefined) {
      init.headers = this.headers({ "Content-Type": "application/json" });
      init.body = JSON.stringify(body);
    }
    const response = await this.fetcher(`${this.base}/v1${path}`, init);
    if (!response.ok) throw await this.failure(response);
    return (await response.json()) as T;
  }

  get<T>(path: string, params: Record<string, string | string[] | undefined> = {}): Promise<T> {
    return this.request<T>("GET", path + query(params));
  }

  /** Every item of a paged list. */
  async all<T>(path: string, params: Record<string, string | string[] | undefined> = {}): Promise<T[]> {
    const items: T[] = [];
    let cursor: string | undefined;
    for (;;) {
      const page = await this.get<Page<T>>(path, { ...params, limit: "500", cursor });
      items.push(...page.items);
      if (page.next_cursor === null) return items;
      cursor = page.next_cursor;
    }
  }

  health = () => this.fetcher(`${this.base}/v1/health`).then((r) => r.ok);
  /** Whether the token is accepted (a cheap authenticated call). */
  check = () => this.get<Page<JobRow>>("/jobs", { limit: "1" }).then(() => true);

  jobs = () => this.all<JobRow>("/jobs");
  job = (id: string) => this.get<JobStatus>(`/jobs/${enc(id)}`);
  events = (id: string, since = 0) =>
    this.get<{ events: ApexEvent[]; next_since: number }>(`/jobs/${enc(id)}/events`, {
      since: String(since),
    });
  stop = (id: string) => this.request<JobStatus>("POST", `/jobs/${enc(id)}:stop`);
  /** What deleting one's own submission removes; nothing is deleted. */
  deletion = (target: DeletionTarget) =>
    this.get<DeletionPlan>("/deletions", target as Record<string, string>);
  /** Delete it, exactly as the plan's token describes. */
  deleteConfirmed = (job: string, token: string) =>
    this.request<{ deleted: string; jobs: number; runs: number; bytes: number }>("POST", "/deletions", {
      job,
      token,
    });
  resume = (id: string) => this.request<JobHandle>("POST", `/jobs/${enc(id)}:resume`, {});

  runs = (annotations: Record<string, string> = {}) =>
    this.all<RunRow>("/runs", {
      annotation: Object.entries(annotations).map(([k, v]) => `${k}=${v}`),
    });
  run = (id: string) => this.get<RunSummary>(`/runs/${enc(id)}`);
  steps = (id: string) => this.get<RunSteps>(`/runs/${enc(id)}/steps`);
  figures = (id: string) => this.get<string[]>(`/runs/${enc(id)}/figures`);
  /** A run's file as a blob (the token cannot ride on an <img src>). */
  async artifact(id: string, path: string): Promise<Blob> {
    const url = `${this.base}/v1/runs/${enc(id)}/artifacts/${path.split("/").map(enc).join("/")}`;
    const response = await this.fetcher(url, { headers: this.headers() });
    if (!response.ok) throw await this.failure(response);
    return response.blob();
  }

  study = (name: string) => this.get<JobStatus>(`/studies/${enc(name)}`);
  comparison = (name: string) => this.get<Comparison>(`/studies/${enc(name)}/comparison`);
  prediction = (path: string) => this.get<PredictionSummary>("/predictions/summary", { path });
  plugins = () => this.get<PluginInfo[]>("/plugins");
  schema = (kind: "history_match" | "prediction" | "study") =>
    this.get<Record<string, unknown>>(`/schemas/${kind}`);

  /** The roots: names (the prefix of every path a route takes) and, for the owner, folders. */
  workspace = () => this.get<{ roots: { name: string; path?: string }[] }>("/workspace");
  files = (path?: string) => this.get<FileEntry[]>("/files", { path });
  readFile = (path: string) => this.get<FileContent>("/files/content", { path });
  saveFile = (path: string, text: string, sha256: string | null) =>
    this.request<{ path: string; sha256: string }>("PUT", "/files/content", { path, text, sha256 });

  whoami = () =>
    this.get<{
      role: "owner" | "ui";
      guest?: string;
      actions: string[];
      ui_folder?: string | null;
      /** When a web-UI token expires (ISO 8601); null for one that does not; absent from older services. */
      expires?: string | null;
    }>("/whoami");
  /** Delete a file of one's own UI folder, if it is still what was read. */
  deleteFile = (path: string, sha256: string) =>
    this.request<{ deleted: string }>("DELETE", "/files/content" + query({ path, sha256 }));
  agents = () => this.get<AgentInfo[]>("/agents");
  /** The web-UI pages installed packages add that this user may see. */
  uiPages = () => this.get<{ pages: ContributedPage[] }>("/ui/pages");
  /** The agents as routes: their rules, or with ``model`` also the model. */
  agentMonitor = (run: string, model = false) =>
    this.request<AgentAnswer & { report: MonitorReport }>("POST", "/agents/monitor", { run, model });
  agentDiagnose = (run: string, model = false) =>
    this.request<AgentAnswer & { diagnosis: Diagnosis }>("POST", "/agents/diagnose", { run, model });
  /** The diagnosis of an optimization's directory: causes and recommended changes. */
  agentDiagnoseOptimization = (optimization: string, model = false) =>
    this.request<AgentAnswer & { diagnosis: Diagnosis }>("POST", "/agents/diagnose", { optimization, model });
  /** The run monitor on an optimization's directory; advisory, it stops nothing. */
  agentMonitorOptimization = (optimization: string, model = false) =>
    this.request<AgentAnswer & { report: MonitorReport }>("POST", "/agents/monitor", { optimization, model });
  agentReport = (target: { run: string } | { study: string } | { optimization: string }, model = false) =>
    this.request<AgentAnswer & { markdown: string }>("POST", "/agents/report", { ...target, model });
  /** The plotter: figures into the run's report/plots, shown by `runFigures`. */
  agentPlot = (run: string, request: string | undefined, model = false) =>
    this.request<AgentAnswer & PlotResult>("POST", "/agents/plot", { run, request: request || undefined, model });
  /** How the run's background drawing stands: none, drawing, done or failed. */
  plotState = (run: string) =>
    this.get<{ state: string; written?: number; error?: string }>(`/agents/plot/${enc(run)}`);
  agentPlan = (plan: { case: string; goal: string; budget?: number; model?: boolean }) =>
    this.request<AgentAnswer & PlannedStudy>("POST", "/agents/plan", plan);
  /** The case-converter: a deck as an APEX case, or a synthetic case. */
  agentConvert = (body: { source: string; history?: string; request?: string; name?: string; model?: boolean }) =>
    this.request<AgentAnswer & ConvertedCase>("POST", "/agents/convert", body);
  agentCreateCase = (body: { request?: string; name?: string; observe?: boolean; model?: boolean }) =>
    this.request<AgentAnswer & CreatedCase>("POST", "/agents/convert/create", body);
  /** The geomodeler: a prior for a configuration's fields, beside it. */
  agentGeomodel = (body: { config: string; request?: string; reference?: string; model?: boolean }) =>
    this.request<AgentAnswer & Geomodeled>("POST", "/agents/geomodel", body);
  chats = () => this.get<Chat[]>("/chats");
  chatModels = () => this.get<ChatModels>("/chats/models");
  createChat = (title: string, context: Record<string, string> = {}) =>
    this.request<Chat>("POST", "/chats", { title, context });
  chat = (id: string) =>
    this.get<Chat & { messages: ChatMessage[]; proposals: Proposal[] }>(`/chats/${enc(id)}`);
  launched = (id: string, n: number, job: string) =>
    this.request<Proposal>("POST", `/chats/${enc(id)}/proposals/${n}:launched`, { job });
  /** A launch answered with another submission's job (not this user's to see): settled with no job. */
  launchedElsewhere = (id: string, n: number) =>
    this.request<Proposal>("POST", `/chats/${enc(id)}/proposals/${n}:launched`, { elsewhere: true });
  discard = (id: string, n: number) => this.request<Proposal>("POST", `/chats/${enc(id)}/proposals/${n}:discard`);
  deleteChat = async (id: string) => {
    const response = await this.fetcher(`${this.base}/v1/chats/${enc(id)}`, {
      method: "DELETE",
      headers: this.headers(),
    });
    if (!response.ok) throw await this.failure(response);
  };
  /** Stop the chat's running answer at once; false if none ran. */
  stopChat = (id: string) => this.request<{ chat: string; stopped: boolean }>("POST", `/chats/${enc(id)}:stop`);
  sendMessage = (id: string, text: string, context: Record<string, string>, model?: string | null) =>
    this.request<{ chat: string }>("POST", `/chats/${enc(id)}/messages`, {
      text,
      context,
      ...(model ? { model } : {}),
    });

  validateText = (text: string, baseDir: string) =>
    this.request<Validation>("POST", "/configs:validate", { config_text: text, base_dir: baseDir });
  submitRun = (configFile: string, approvalToken?: string) =>
    this.request<JobHandle>("POST", "/runs", {
      config_file: configFile,
      client: "ui",
      approval_token: approvalToken ?? null,
    });
  submitStudy = (studyFile: string, approvalToken?: string) =>
    this.request<JobHandle>("POST", "/studies", {
      study_file: studyFile,
      client: "ui",
      approval_token: approvalToken ?? null,
    });
  optimizations = () => this.get<OptimizationRow[]>("/optimizations");
  optimizationSummary = (path: string) => this.get<OptimizationSummary>("/optimizations/summary", { path });
  submitOptimization = (optimizationFile: string, approvalToken?: string) =>
    this.request<JobHandle>("POST", "/optimizations", {
      optimization_file: optimizationFile,
      client: "ui",
      approval_token: approvalToken ?? null,
    });
  examples = () => this.get<{ examples: Example[] }>("/examples");
  copyExample = (id: string) => this.request<CopiedExample>("POST", "/examples:copy", { id });
  submitPrediction = (predictionFile: string) =>
    this.request<JobHandle>("POST", "/predictions", { prediction_file: predictionFile, client: "ui" });
}

function enc(value: string): string {
  return encodeURIComponent(value);
}

export function query(params: Record<string, string | string[] | undefined>): string {
  const search = new URLSearchParams();
  for (const [key, value] of Object.entries(params)) {
    if (value === undefined) continue;
    for (const item of Array.isArray(value) ? value : [value]) search.append(key, item);
  }
  const text = search.toString();
  return text ? `?${text}` : "";
}

async function problemOf(response: Response): Promise<Problem> {
  try {
    const body = (await response.json()) as Partial<Problem>;
    return {
      type: body.type ?? "about:blank",
      title: body.title ?? response.statusText,
      status: body.status ?? response.status,
      detail: body.detail ?? response.statusText,
      ...body,
    };
  } catch {
    return { type: "about:blank", title: response.statusText, status: response.status, detail: "" };
  }
}
