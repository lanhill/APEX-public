import { describe, expect, it } from "vitest";

import { Api, ApiError, query, type Fetch } from "./api";
import {
  anyOtherEngine,
  comparisonMetricLabel,
  difference,
  engineOf,
  studyCells,
  studyNames,
  toggled,
} from "./comparison";
import {
  chatModel,
  chosenModel,
  modelLabel,
  openingModel,
  pendingCards,
  stepText,
  streamed,
  switchNote,
} from "./components/Chat";
import { answeredBy } from "./components/Agents";
import { contributedLinks } from "./contributed";
import { deletionLines, sizeText } from "./components/Delete";
import { fmt, metricLabels,
  stepNumbers, niceTicks, scale, series } from "./chart";
import {
  examplesByWorkflow,
  frontMembers,
  hypervolumePoints,
  frontInfeasible,
  isMultiObjective,
  maximisedObjectives,
  objectiveBests,
  parallelRows,
} from "./front";
import { annotationItems, isNotFound, noFigures, reusedRun, submittedNote } from "./runs";
import { tickDigits } from "./components/MetricChart";
import { href, parse } from "./router";
import { SseParser } from "./sse";
import { parse as parseMarkdown, inline } from "./markdown";
import { makePaths } from "./paths";
import { assumptionsOf, launchesOptimization, launchKind, unmatchedOf } from "./proposals";
import { withoutAdr } from "./text";
import {
  clearSession,
  detailOf,
  endSession,
  forgetSaved,
  loadSession,
  nameMatches,
  refusal,
  refusalHandler,
  refusalText,
  REMEMBER_DAYS,
  rememberUntil,
  savedSignIn,
  savedUntil,
  saveSession,
  tokenName,
  type Stores,
} from "./session";
import {
  bestControls,
  feasibility,
  historyColumns,
  historyValue,
  objectiveChart,
  objectiveSense,
  optimizationLabel,
  objectiveQuantity,
  plainNumber,
  unscaled,
} from "./optimization";
import { agentLabel, visibilityOf, visibleAgents, whereInUi } from "./agents";
import { filterGroup, groupPlugins, optionRows, registryLabel, summaryParts, typeText } from "./plugins";
import { dirname, isReadOnly, kindOf, syntaxError } from "./toml";
import type { OptimizationStep, Proposal, StepStatistics } from "./api";

const row = (metric: string, label: string, iteration: number, mean: number | null): StepStatistics => ({
  metric,
  label,
  iteration,
  mean,
  std: 1,
  median: mean,
  p10: mean === null ? null : mean - 1,
  p90: mean === null ? null : mean + 1,
});

describe("SSE parser", () => {
  it("splits messages across chunks, skips comments and keeps the last id", () => {
    const parser = new SseParser();
    expect(parser.push(": keep-alive\n\nid: 3\nevent: forecast_done\nda")).toEqual([]);
    const out = parser.push('ta: {"index": 3}\n\nid: 4\ndata: a\ndata: b\r\n\r\n');
    expect(out).toEqual([
      { id: "3", event: "forecast_done", data: '{"index": 3}' },
      { id: "4", event: "message", data: "a\nb" },
    ]);
  });
});

describe("chart helpers", () => {
  it("give tick labels enough digits to tell neighbours apart", () => {
    expect(tickDigits(niceTicks(-0.2, 0.2))).toBe(2);
    expect(tickDigits([90, 95, 100])).toBe(3);
    const large = niceTicks(1e6, 1e6 + 10);
    expect(new Set(large.map((t) => fmt(t, tickDigits(large)))).size).toBe(large.length);
    const narrow = niceTicks(-225.82, -225.6);
    const digits = tickDigits(narrow);
    expect(new Set(narrow.map((t) => fmt(t, digits))).size).toBe(narrow.length);
    expect(tickDigits([0, 10, 20])).toBe(2);
    expect(tickDigits([5])).toBe(2);
  });

  it("pad a scale so the extremes sit off the frame, and not by default", () => {
    const plain = scale([0, 10], 100, 0);
    expect([plain(0), plain(10)]).toEqual([100, 0]);
    const padded = scale([0, 10], 100, 0, true, 0.05);
    expect(padded(0)).toBeLessThan(100);
    expect(padded(10)).toBeGreaterThan(0);
    expect(padded(5)).toBeCloseTo(50);
    // On a log scale the margin is in decades: 1 and 1000 stay symmetric about 10^1.5.
    const log = scale([1, 1000], 100, 0, true, 0.1);
    expect(log.log).toBe(true);
    expect(100 - log(1)).toBeCloseTo(log(1000));
  });

  it("numbers the steps without the rejected trials", () => {
    // Records 0..5 with 2 and 3 rejected: the accepted ones are steps 0, 1, 2, 3.
    const stepOf = stepNumbers([{ iteration: 0 }, { iteration: 1 }, { iteration: 4 }, { iteration: 5, step: 3 }]);
    const rows = [row("rmse", "all", 5, 1), row("rmse", "all", 4, 2), row("rmse", "all", 2, 9), row("rmse", "all", 0, 4)];
    expect(series(rows, "rmse", "all", stepOf).map((p) => [p.x, p.mean])).toEqual([
      [0, 4],
      [2, 2],
      [3, 1],
    ]);
  });

  it("orders a metric's steps and leaves out missing means", () => {
    const rows = [row("rmse", "PERMX", 2, 3), row("rmse", "PERMX", 0, 5), row("rmse", "PERMX", 1, null), row("x", "all", 0, 1)];
    expect(series(rows, "rmse", "PERMX").map((p) => [p.x, p.mean, p.low])).toEqual([
      [0, 5, 4],
      [2, 3, 2],
    ]);
    expect([...metricLabels(rows)]).toEqual([
      ["rmse", ["PERMX"]],
      ["x", ["all"]],
    ]);
    const mismatches = [
      row("data_mismatch", "all", 0, 9),
      row("data_mismatch", "WOPR", 0, 5),
      row("normalized_data_mismatch", "all", 0, 1),
      row("normalized_data_mismatch", "WOPR", 0, 2),
      row("cv_normalized_data_mismatch", "WBHP", 0, 2),
      row("rmse", "PERMX", 0, 1),
    ];
    // The data mismatches overall only; other metrics per label.
    expect([...metricLabels(mismatches)]).toEqual([
      ["data_mismatch", ["all"]],
      ["normalized_data_mismatch", ["all"]],
      ["rmse", ["PERMX"]],
    ]);
  });

  it("uses a log scale only for positive values over two decades", () => {
    const log = scale([1, 1000], 0, 100);
    expect(log.log).toBe(true);
    expect(log(10)).toBeCloseTo(100 / 3);
    expect(log.ticks).toEqual([1, 10, 100, 1000]);
    const linear = scale([-1, 1000], 0, 100);
    expect(linear.log).toBe(false);
    expect(linear(-1)).toBe(0);
    expect(scale([5, 5], 0, 10)(5)).toBe(5); // a flat series sits in the middle
  });

  it("chooses round ticks and formats numbers compactly", () => {
    expect(niceTicks(0, 1)).toEqual([0, 0.2, 0.4, 0.6, 0.8, 1]);
    expect(fmt(null)).toBe("–");
    expect(fmt(Number.NaN)).toBe("–");
    expect(fmt(1234567)).toBe("1.23e+6");
    expect(fmt(0.5)).toBe("0.5");
  });
});

describe("routes", () => {
  it("round-trip through the hash", () => {
    const routes = [
      { page: "job", id: "a b" },
      { page: "run", id: "0123" },
      { page: "runs", filter: "study=x" },
      { page: "study", name: "loc" },
      { page: "prediction", path: "/r/predictions/p&1" },
      { page: "editor", path: "/c/run.toml" },
      { page: "plugins" },
      { page: "plugins", workflow: "optimization" },
      { page: "plugins", workflow: "history_matching", registry: "localizations", name: "adaptive_prior" },
      { page: "agents" },
      { page: "optimizations" },
      { page: "optimization", path: "/w/UI/owner/optimizations/npv/ab12&c" },
      { page: "agents", name: "monitor" },
    ] as const;
    for (const route of routes) expect(parse(href(route))).toEqual(route);
    expect(parse("")).toEqual({ page: "jobs" });
  });
});

describe("TOML helpers", () => {
  it("tell a study from a run and find syntax errors", () => {
    expect(kindOf({ study: {}, x: 1 })).toBe("study");
    expect(kindOf({ prediction: {} })).toBe("prediction");
    expect(kindOf({ run: {}, study: {} })).toBe("history_match");
    // As the service decides (ADR 0043): both tables, wherever they are.
    expect(kindOf({ controls: {}, optimization: {}, optimizer: {} })).toBe("optimization");
    expect(kindOf({ optimization: {} })).toBe("history_match");
    expect(syntaxError('[run]\nname = "a"\n')).toBeNull();
    expect(syntaxError("[run\n")).not.toBeNull();
    expect(dirname("/a/b/c.toml")).toBe("/a/b");
    expect(dirname("/c.toml")).toBe("/");
  });
});

describe("API client", () => {
  it("sends the token, repeats list parameters and follows cursors", async () => {
    const calls: { url: string; init?: RequestInit }[] = [];
    const pages: Record<string, unknown> = {
      "": { items: [{ run_id: "a" }], next_cursor: "Mg==" },
      "Mg==": { items: [{ run_id: "b" }], next_cursor: null },
    };
    const fetcher: Fetch = async (url, init) => {
      calls.push({ url, init });
      const cursor = new URL(url, "http://x").searchParams.get("cursor") ?? "";
      return new Response(JSON.stringify(pages[cursor]), { status: 200 });
    };
    const api = new Api("secret", "", fetcher);
    const runs = await api.runs({ study: "s", case: "c" });
    expect(runs.map((r) => r.run_id)).toEqual(["a", "b"]);
    expect(calls[0]?.url).toBe("/v1/runs?annotation=study%3Ds&annotation=case%3Dc&limit=500");
    expect((calls[0]?.init?.headers as Record<string, string>).Authorization).toBe("Bearer secret");
    // The web UI names itself, so the owner sees only the owner's work there (ADR 0028 r2).
    expect((calls[0]?.init?.headers as Record<string, string>)["X-APEX-Client"]).toBe("ui");
    expect(query({ a: undefined, b: ["1", "2"] })).toBe("?b=1&b=2");
  });

  it("turns problem documents into errors carrying the approval token", async () => {
    const fetcher: Fetch = async () =>
      new Response(JSON.stringify({ title: "Approval required", status: 403, detail: "too big", token: "abc" }), {
        status: 403,
      });
    const error = await new Api("t", "", fetcher).submitRun("/c/run.toml").catch((e: unknown) => e);
    expect(error).toBeInstanceOf(ApiError);
    expect((error as ApiError).problem.token).toBe("abc");
    expect((error as ApiError).message).toBe("too big");
  });
});

describe("workspace paths", () => {
  it("send one root's relative paths as they are, and show its server paths relative", () => {
    const paths = makePaths([{ name: "", path: "/srv/cases/APEX" }]);
    // Every route takes workspace-relative paths (finding 31): no server path is sent.
    expect(paths.toServer("m5/configs/gies.toml")).toBe("m5/configs/gies.toml");
    expect(paths.toServer("m5/configs/")).toBe("m5/configs");
    // An absolute path (the owner's) is sent as it is.
    expect(paths.toServer("/srv/cases/APEX/m5/a.toml")).toBe("/srv/cases/APEX/m5/a.toml");
    expect(paths.folderOf("/srv/cases/APEX/m5/a.toml")).toBe("/srv/cases/APEX/m5");
    expect(paths.toServer("")).toBe(".");
    expect(paths.folderOf("m5/configs/gies.toml")).toBe("m5/configs");
    expect(paths.folderOf("top.toml")).toBe(".");
    expect(paths.display("no file /srv/cases/APEX/m5/obs.csv here")).toBe("no file m5/obs.csv here");
    expect(paths.display("/srv/cases/APEX")).toBe("(workspace)");
    expect(paths.display("/srv/cases/APEXX/x")).toBe("/srv/cases/APEXX/x"); // another folder
    expect(paths.display("/home/u/.apex/service/runs/r")).toBe("/home/u/.apex/service/runs/r");
  });

  it("prefix several roots by name, the innermost first", () => {
    const paths = makePaths([
      { name: "cases", path: "/data/cases" },
      { name: "deep", path: "/data/cases/deep" },
    ]);
    expect(paths.toServer("deep/a.toml")).toBe("deep/a.toml");
    expect(paths.folderOf("deep/sub/a.toml")).toBe("deep/sub");
    expect(paths.display("/data/cases/deep/a.toml and /data/cases/b.toml")).toBe(
      "deep/a.toml and cases/b.toml",
    );
    expect(() => paths.toServer("other/a.toml")).toThrow();
  });

  it("work from names alone, as a guest's workspace gives them", () => {
    const one = makePaths([{ name: "" }]);
    expect(one.toServer("UI/guests/ann/drafts/c1/study.toml")).toBe("UI/guests/ann/drafts/c1/study.toml");
    expect(one.folderOf("UI/guests/ann/a.toml")).toBe("UI/guests/ann");
    // The server sends guests relative paths already: nothing to rewrite.
    expect(one.display("UI/guests/ann/runs/r1 failed")).toBe("UI/guests/ann/runs/r1 failed");
    const two = makePaths([{ name: "cases" }, { name: "more" }]);
    expect(two.toServer("more/x.toml")).toBe("more/x.toml");
    expect(two.folderOf("cases/x.toml")).toBe("cases");
    expect(() => two.toServer("x.toml")).toThrow();
    expect(two.path("/abs/elsewhere")).toBe("/abs/elsewhere");
  });
});

describe("the assistant's Markdown", () => {
  it("read a report's table into rows of cells, and leave other pipes alone", () => {
    const blocks = parseMarkdown("## Key numbers\n\n| Quantity | Value |\n|---|---:|\n| **iterations** | 4 |\n| rmse | `0.12` |\n\nAfter.");
    const table = blocks.find((b) => b.kind === "table");
    expect(table).toBeDefined();
    if (table?.kind !== "table") return;
    expect(table.header.map((c) => c.map((s) => s.text).join(""))).toEqual(["Quantity", "Value"]);
    expect(table.rows.length).toBe(2);
    expect(table.rows[0]![0]![0]).toEqual({ kind: "bold", text: "iterations" });
    expect(table.rows[1]![1]![0]).toEqual({ kind: "code", text: "0.12" });
    expect(blocks[blocks.length - 1]).toMatchObject({ kind: "paragraph" });
    // A line with pipes but no separator row is not a table.
    expect(parseMarkdown("a | b\nc | d").map((b) => b.kind)).toEqual(["paragraph"]);
    // Nor is a lone "| a" before a "---" rule (the separator has the header's cell count).
    expect(parseMarkdown("| a\n---").some((b) => b.kind === "table")).toBe(false);
  });

  it("reads blocks and inline marks, never HTML", () => {
    const blocks = parseMarkdown(
      "# Result\nThe **RMSE** fell to `0.2`.\n\n- one\n- two\n\n1. first\n\n```\ncode <b>\n```\n<script>x</script>",
    );
    expect(blocks.map((b) => b.kind)).toEqual(["heading", "paragraph", "list", "list", "code", "paragraph"]);
    expect(inline("a **b** `c` *d*")).toEqual([
      { kind: "text", text: "a " },
      { kind: "bold", text: "b" },
      { kind: "text", text: " " },
      { kind: "code", text: "c" },
      { kind: "text", text: " " },
      { kind: "italic", text: "d" },
    ]);
    const last = blocks[blocks.length - 1];
    expect(last?.kind === "paragraph" && last.inline[0]?.text).toBe("<script>x</script>"); // text, not markup
  });
});

describe("study comparisons", () => {
  const d = (method: string | null, variant: string, refMethod: string | null, ref: string, mean: number) => ({
    method,
    variant,
    reference_method: refMethod,
    reference: ref,
    n: 2,
    mean_difference: mean,
    standard_error: null,
    p_value: null,
    note: "",
  });
  const stats = (method: string | null, variant: string) => ({
    method,
    variant,
    n: 2,
    mean: 1,
    std: null,
    minimum: 1,
    maximum: 1,
    values: [1, 1],
  });

  it("find a group's difference within its method and against the reference method", () => {
    const differences = [
      d("esmda", "adaptive", "esmda", "plain", 1),
      d("gies", "adaptive", "gies", "plain", 2),
      d("gies", "plain", "esmda", "plain", 3),
      d("gies", "adaptive", "esmda", "adaptive", 4),
    ];
    const gies = stats("gies", "adaptive");
    expect(difference(differences, gies, { method: "gies", variant: "plain" })?.mean_difference).toBe(2);
    expect(difference(differences, gies, { method: "esmda", variant: "adaptive" })?.mean_difference).toBe(4);
    expect(difference(differences, stats("esmda", "plain"), { method: "esmda", variant: "plain" })).toBeUndefined();
  });

  it("match studies without methods, whose comparisons carry none", () => {
    const old = [{ ...d(null, "two", null, "four", 5), method: undefined, reference_method: undefined }];
    const v = { ...stats(null, "two"), method: undefined };
    expect(difference(old, v, { method: null, variant: "four" })?.mean_difference).toBe(5);
  });
});

describe("the assistant's steps", () => {
  it("say why a tool refused, rather than that it failed", () => {
    const event = (data: Record<string, unknown>) => ({ n: 1, created: "", kind: "tool_result", data });
    expect(stepText(event({ tool: "plan_study", ok: true }))).toBe("plan_study answered");
    expect(stepText(event({ tool: "plan_study", ok: false, error: "over the budget" }))).toBe(
      "plan_study refused: over the budget",
    );
    expect(stepText(event({ tool: "plan_study", ok: false }))).toBe("plan_study refused the request");
  });
});

describe("deleting one's own work", () => {
  it("say that deleting a study's cell deletes the whole study", () => {
    const plan = {
      job: "study-job", kind: "study", owner: "guest:alice", jobs: ["cell-0", "study-job", "cell-1"], runs: [],
      dirs: ["UI/guests/alice/studies/optst/toy/base/r0", "UI/guests/alice/studies/optst/toy/base/r1"],
      studies: ["optst"], kept: [], size: 2000, blocked: [], token: "t", optimizations: ["/w/r0", "/w/r1"],
    };
    expect(deletionLines(plan, { job: "cell-0" })[0]).toBe("This is a cell of study optst: the whole study goes, with its 2 optimizations.");
    // Asked about the study itself: no such line.
    expect(deletionLines(plan, { study: "optst" })[0]).not.toMatch(/cell of study/);
    expect(deletionLines(plan)[0]).not.toMatch(/cell of study/);
  });

  it("lists what the confirmation deletes", () => {
    const plan = {
      job: "20261001T1", kind: "study", owner: "guest:alice", jobs: ["a", "b"], runs: ["r"],
      dirs: ["UI/guests/alice/studies/s/c/v/r0"], studies: ["s"], kept: [], size: 2_500_000,
      blocked: [], token: "t",
    };
    expect(deletionLines(plan)).toEqual([
      "2 jobs (study 20261001T1)",
      "folder UI/guests/alice/studies/s/c/v/r0",
      "study s (its record and comparison)",
      "2.5 MB on disk",
    ]);
    expect(sizeText(400)).toBe("1 kB");
    expect(sizeText(250_000_000)).toBe("250 MB");
  });
});

describe("deleting a file of one's own folder", () => {
  it("sends the path and the SHA-256 the editor read", async () => {
    const calls: [string, RequestInit | undefined][] = [];
    const fetcher: Fetch = async (url, init) => {
      calls.push([String(url), init]);
      return new Response(JSON.stringify({ deleted: "UI/owner/drafts/c/x.toml" }), { status: 200 });
    };
    const api = new Api("tok", "http://h", fetcher);
    expect(await api.deleteFile("UI/owner/drafts/c/x.toml", "abc")).toEqual({ deleted: "UI/owner/drafts/c/x.toml" });
    const [url, init] = calls[0] ?? ["", undefined];
    expect(init?.method).toBe("DELETE");
    expect(url).toBe("http://h/v1/files/content?path=UI%2Fowner%2Fdrafts%2Fc%2Fx.toml&sha256=abc");
  });
});

describe("a streamed answer", () => {
  it("grows by its pieces and starts again with each model call", () => {
    const event = (kind: string, data: Record<string, unknown> = {}) => ({ n: 1, created: "", kind, data });
    let draft = streamed("", event("answer_delta", { text: "The run " }));
    draft = streamed(draft, event("answer_delta", { text: "converged." }));
    expect(draft).toBe("The run converged.");
    expect(streamed(draft, event("tool_call", { tool: "run_summary" }))).toBe("The run converged.");
    expect(streamed(draft, event("model_call", { turn: 2 }))).toBe("");
  });
});

describe("the agents' buttons", () => {
  it("say who answered, and why the model did not", () => {
    expect(answeredBy({ decided_by: "model", reason: null })).toBe("the model");
    expect(answeredBy({ decided_by: "rules", reason: "no model (no key): the rules answered" })).toBe(
      "the agent's rules (no model (no key): the rules answered)",
    );
  });

  it("ask the agent routes, with the model only when asked", async () => {
    const calls: [string, RequestInit | undefined][] = [];
    const fetcher: Fetch = async (url, init) => {
      calls.push([String(url), init]);
      return new Response(JSON.stringify({ decided_by: "rules", reason: null, markdown: "# r" }), { status: 200 });
    };
    const api = new Api("tok", "http://h", fetcher);
    await api.agentReport({ study: "s" });
    await api.agentDiagnose("run1", true);
    expect(calls.map(([url]) => url)).toEqual(["http://h/v1/agents/report", "http://h/v1/agents/diagnose"]);
    expect(JSON.parse(String(calls[0]?.[1]?.body))).toEqual({ study: "s", model: false });
    expect(JSON.parse(String(calls[1]?.[1]?.body))).toEqual({ run: "run1", model: true });
  });

  it("ask the case-converter and the geomodeler for drafts", async () => {
    const calls: [string, RequestInit | undefined][] = [];
    const fetcher: Fetch = async (url, init) => {
      calls.push([String(url), init]);
      return new Response(JSON.stringify({ decided_by: "rules", reason: null }), { status: 200 });
    };
    const api = new Api("tok", "http://h", fetcher);
    await api.agentConvert({ source: "decks/CASE.DATA", name: "small" });
    await api.agentCreateCase({ name: "box", observe: true });
    await api.agentGeomodel({ config: "UI/owner/drafts/x/config.toml" });
    expect(calls.map(([url]) => url)).toEqual([
      "http://h/v1/agents/convert",
      "http://h/v1/agents/convert/create",
      "http://h/v1/agents/geomodel",
    ]);
    expect(JSON.parse(String(calls[1]?.[1]?.body))).toEqual({ name: "box", observe: true });
    expect(whereInUi("geomodeler")?.pages).toEqual(["editor"]);
    expect(agentLabel("case-converter")).toBe("Case converter");
  });

});

describe("engines in studies (ADR 0041)", () => {
  it("read a missing engine as APEX's and show a column only for another engine", () => {
    expect(engineOf({})).toBe("apex");
    expect(engineOf({ engine: "ert" })).toBe("ert");
    expect(anyOtherEngine([{}, { engine: "apex" }])).toBe(false);
    expect(anyOtherEngine([{ engine: "apex" }, { engine: "ert" }])).toBe(true);
    expect(anyOtherEngine([])).toBe(false);
  });

  it("list a study's cells from its runs' annotations, ERT's marked by the catalogue", () => {
    const run = (id: string, annotations: Record<string, string>) => ({ run_id: id, name: id, annotations });
    const cells = studyCells(
      [
        run("b", { study: "s", case: "m5", method: "ert_esmda", engine: "ert", variant: "plain", replicate: "1" }),
        run("a", { study: "s", case: "m5", method: "esmda", variant: "plain", replicate: "1" }),
        run("c", { study: "s", case: "m5", method: "esmda", variant: "plain", replicate: "0" }),
        run("x", { study: "other", case: "m5", variant: "plain", replicate: "0" }),
      ],
      "s",
    );
    expect(cells.map((c) => c.key)).toEqual(["b", "c", "a"]);
    expect(cells.every((c) => c.kind === "run" && c.target === c.key)).toBe(true);
    expect(cells.map((c) => c.engine)).toEqual(["ert", "apex", "apex"]);
    expect(cells[1]).toMatchObject({ case: "m5", method: "esmda", variant: "plain", replicate: 0 });
  });

  it("keep cells without a method (studies without [study.methods])", () => {
    const [cell] = studyCells([{ run_id: "r", annotations: { study: "s", case: "c", variant: "v", replicate: "2" } }], "s");
    expect(cell).toMatchObject({ method: null, engine: "apex", replicate: 2 });
  });

  it("list a study of optimizations' cells, each by its newest job, linked to its directory", () => {
    const opt = (job: string, path: string, created: string, replicate: string) => ({
      job,
      path,
      created,
      annotations: { submitted_by: "guest:alice", study: "optst", case: "toy", variant: "base", replicate },
    });
    const cells = studyCells([], "optst", [
      opt("j1", "/w/UI/guests/alice/studies/optst/toy/base/r1", "2026-10-04T14:48:45", "1"),
      opt("j0", "/w/UI/guests/alice/studies/optst/toy/base/r0", "2026-10-04T14:48:45", "0"),
      opt("j0b", "/w/UI/guests/alice/studies/optst/toy/base/r0", "2026-10-04T15:00:00", "0"), // a resume
      { ...opt("x", "/w/other", "2026-10-04", "0"), annotations: { study: "another" } },
    ]);
    expect(cells.map((c) => [c.kind, c.key, c.replicate])).toEqual([
      ["optimization", "j0b", 0],
      ["optimization", "j1", 1],
    ]);
    expect(cells[0]!.target).toBe("/w/UI/guests/alice/studies/optst/toy/base/r0");
  });

  it("find studies by runs and by optimizations, counting each optimization once", () => {
    const studies = studyNames(
      [{ annotations: { study: "hm" } }, { annotations: { study: "hm" } }, { annotations: {} }],
      [
        { path: "/a/r0", annotations: { study: "optst" } },
        { path: "/a/r0", annotations: { study: "optst" } }, // its resume
        { path: "/a/r1", annotations: { study: "optst" } },
      ],
    );
    expect(studies).toEqual([
      { name: "hm", runs: 2, optimizations: 0 },
      { name: "optst", runs: 0, optimizations: 2 },
    ]);
  });

  it("name an optimization study's metrics for a person, and leave a run's as they are", () => {
    expect(comparisonMetricLabel("objective:final")).toBe("objective at the end (minimised)");
    expect(comparisonMetricLabel("evaluations:all")).toBe("evaluations");
    expect(comparisonMetricLabel("rmse:all")).toBe("rmse:all");
    expect(comparisonMetricLabel("constructor")).toBe("constructor");
  });
});

describe("collapsible plugin registries", () => {
  it("toggle a registry without changing the set it came from", () => {
    const none = new Set<string>();
    const one = toggled(none, "methods");
    expect([...one]).toEqual(["methods"]);
    expect(none.size).toBe(0);
    expect([...toggled(one, "methods")]).toEqual([]);
    expect([...toggled(one, "priors")].sort()).toEqual(["methods", "priors"]);
  });
});

describe("the Plugins page", () => {
  const plugin = (registry: string, name: string, summary = "", options: Record<string, unknown> = {}) => ({
    registry,
    name,
    status: null,
    summary,
    options,
    supplied: [],
  });
  const all = [
    plugin("simulators", "opm_flow"),
    plugin("localizations", "adaptive_prior", "Adaptive localization, prior only"),
    plugin("methods", "esmda", "ES-MDA"),
    plugin("objectives", "rosenbrock"),
    plugin("engines", "ert"),
    plugin("methods", "gies", "GIES"),
  ];

  it("group registries by workflow, in the workflows' order, with unknown ones under Other", () => {
    const groups = groupPlugins(all);
    expect(groups.map((g) => g.workflow.key)).toEqual(["history_matching", "optimization", "simulation", "other"]);
    expect(groups[0]!.registries.map((r) => r.registry)).toEqual(["methods", "localizations"]);
    expect(groups[0]!.registries[0]!.plugins.map((p) => p.name)).toEqual(["esmda", "gies"]);
    expect(groups[0]!.count).toBe(3);
    expect(groups[3]!.registries.map((r) => r.registry)).toEqual(["engines"]);
  });

  it("follow each plugin's declared workflow when the service sends one (null: Other)", () => {
    const declared = [
      { ...plugin("metrics", "rmse"), workflow: "history_matching" },
      { ...plugin("metrics", "npv_gap"), workflow: "optimization" },
      { ...plugin("simulators", "opm_flow"), workflow: null },
      { ...plugin("surrogates", "gp"), workflow: "optimization" },
    ];
    const groups = groupPlugins(declared);
    expect(groups.map((g) => g.workflow.key)).toEqual(["history_matching", "optimization", "other"]);
    expect(groups[1]!.registries.map((r) => r.registry)).toEqual(["metrics", "surrogates"]);
    expect(groups[2]!.registries[0]!.plugins.map((p) => p.name)).toEqual(["opm_flow"]);
  });

  it("put an unknown declared workflow under Other, and mix declared and undeclared plugins", () => {
    const groups = groupPlugins([
      { ...plugin("engines", "ert"), workflow: "ert" },
      plugin("methods", "esmda"),
      { ...plugin("metrics", "rmse"), workflow: "history_matching" },
      { ...plugin("metrics", "npv_gap"), workflow: "optimization" },
    ]);
    expect(groups.map((g) => g.workflow.key)).toEqual(["history_matching", "optimization", "other"]);
    expect(groups[0]!.registries.map((r) => r.registry)).toEqual(["methods", "metrics"]);
    // One registry may be split across workflows: each keeps its own plugins.
    expect(groups[1]!.registries[0]!.plugins.map((p) => p.name)).toEqual(["npv_gap"]);
    expect(groups[2]!.registries[0]!.plugins.map((p) => p.name)).toEqual(["ert"]);
  });

  it("search names and summaries, dropping registries without a match", () => {
    const hm = groupPlugins(all)[0]!;
    expect(filterGroup(hm, "  ").map((r) => r.registry)).toEqual(["methods", "localizations"]);
    const found = filterGroup(hm, "PRIOR");
    expect(found.map((r) => r.registry)).toEqual(["localizations"]);
    expect(filterGroup(hm, "gies")[0]!.plugins.map((p) => p.name)).toEqual(["gies"]);
  });

  it("read options from pydantic's schema", () => {
    const rows = optionRows({
      properties: {
        inflation: { items: { type: "number" }, type: "array" },
        taper: { default: "hard", enum: ["hard", "soft"], type: "string" },
        tol: { anyOf: [{ type: "number" }, { type: "null" }], default: null },
        epf: { $ref: "#/$defs/Epf", default: null },
        deck: { anyOf: [{ type: "string" }, { type: "string", format: "path" }], description: "The deck." },
      },
      required: ["inflation", "deck"],
      $defs: { Epf: { anyOf: [{ type: "object" }, { type: "null" }] } },
    });
    expect(rows).toEqual([
      { name: "inflation", type: "list of number", required: true },
      { name: "taper", type: "choice", required: false, default: '"hard"', choices: ['"hard"', '"soft"'] },
      { name: "tol", type: "number or null", required: false, default: "none" },
      { name: "epf", type: "table or null", required: false, default: "none" },
      { name: "deck", type: "string or path", required: true, description: "The deck." },
    ]);
    expect(optionRows({})).toEqual([]);
    expect(typeText({ type: "array" })).toBe("list");
  });

  it("label registries and show reStructuredText code", () => {
    expect(registryLabel("update_strategies")).toBe("Update strategies");
    expect(summaryParts("Taper from ``ini_only`` (PET).")).toEqual([
      { text: "Taper from ", code: false },
      { text: "ini_only", code: true },
      { text: " (PET).", code: false },
    ]);
  });
});

describe("the Agents page", () => {
  const agent = (name: string, visibility?: "public" | "private") => ({
    name,
    summary: "",
    command: `apex agent ${name}`,
    tool: null,
    adr: "0025",
    ...(visibility ? { visibility } : {}),
  });
  const names = (list: { name: string }[]) => list.map((a) => a.name);
  const service = [agent("monitor"), agent("planner", "public"), agent("secret", "private")];

  it("never show private agents, wherever they are listed", () => {
    expect(names(visibleAgents(service))).toEqual(["monitor", "planner"]);
    expect(names(visibleAgents([agent("monitor", "private")]))).toEqual([]);
  });

  it("treat an agent the service does not tag as public", () => {
    expect(visibilityOf(agent("forecaster"))).toBe("public");
  });

  it("look up agents by their own names only (names come from the service)", () => {
    expect(visibilityOf(agent("constructor"))).toBe("public");
    expect(whereInUi("toString")).toBeUndefined();
    expect(whereInUi("report")?.pages).toEqual(["runs", "optimizations", "studies"]);
  });

  it("label agents", () => {
    expect(agentLabel("diagnosis")).toBe("Diagnosis");
    expect(agentLabel("run-monitor")).toBe("Run monitor");
  });
});

describe("optimizations (ADR 0043)", () => {
  const step = (iteration: number, objective: number | null, x?: number[], extra: Record<string, unknown> = {}) => ({
    kind: iteration === 0 ? "start" : "step",
    iteration,
    objective,
    ...(x ? { x } : {}),
    ...extra,
  });
  const summary = (history: OptimizationStep[], result: Record<string, unknown> | null = null) => ({
    path: "/w/optimizations/rosen/ab",
    name: "rosen",
    state: (result ? "finished" : "running") as "finished" | "running",
    optimizer: "enopt",
    objective: "rosenbrock",
    controls: ["x[0]", "x[1]"],
    history,
    result,
    evaluations: 0,
    failed_evaluations: 0,
  });

  it("chart the objective at each row that has one, numbered from the start", () => {
    const chart = objectiveChart([step(0, 10), step(1, null), step(2, 4), step(3, Number.NaN)]);
    expect(chart.points.map((p) => [p.x, p.mean])).toEqual([
      [0, 10],
      [1, 4],
    ]);
    expect(chart).toMatchObject({ restarts: [], penalised: false });
  });

  it("restart the line at each EPF penalty update, whose iteration starts again at 0", () => {
    const epf = { kind: "epf_update", iteration: 0, objective: 9, x: [0.4, 1.4], epf_iteration: 2 };
    const chart = objectiveChart([step(0, 10, [1, 2]), step(1, 5, [0.5, 1.5]), epf, step(1, 7, [0.3, 1.3])]);
    expect(chart.points.map((p) => p.x)).toEqual([0, 1, 2, 3]);
    expect(chart).toMatchObject({ restarts: [2], penalised: true });
  });

  it("take penalised from the service when it says, from the first outer iteration on", () => {
    // EPF configured, still in its first outer iteration: no update row yet, but penalised.
    expect(objectiveChart([step(0, 10), step(1, 5)], true).penalised).toBe(true);
    // The service's word wins over the rows; without it, the rows decide.
    expect(objectiveChart([step(0, 10), { kind: "epf_update", iteration: 0, objective: 9 }], false).penalised).toBe(false);
    expect(objectiveChart([step(0, 10), { kind: "epf_update", iteration: 0, objective: 9 }]).penalised).toBe(true);
  });

  it("show the result's controls once finished, else the latest accepted point", () => {
    const running = summary([step(0, 10, [1, 2]), step(1, 5, [0.5, 1.5]), step(2, 4)]);
    expect(bestControls(running)).toEqual([
      { name: "x[0]", value: 0.5 },
      { name: "x[1]", value: 1.5 },
    ]);
    const done = summary([step(0, 10, [1, 2])], { x: [0.1, 0.2], names: ["a", "b"], objective: 0.3 });
    expect(bestControls(done).map((c) => c.name)).toEqual(["a", "b"]);
    expect(bestControls(summary([]))).toEqual([]);
    // Before any step: the start; a penalty update is not an accepted point.
    expect(bestControls(summary([step(0, 10, [1, 2])])).map((c) => c.value)).toEqual([1, 2]);
    const epf = { kind: "epf_update", iteration: 0, objective: 9, x: [9, 9] };
    expect(bestControls(summary([step(0, 10, [1, 2]), step(1, 5, [0.5, 1.5]), epf])).map((c) => c.value)).toEqual([
      0.5, 1.5,
    ]);
  });

  it("put the shared history columns first, then each optimizer's own, never x", () => {
    const columns = historyColumns([step(0, 1, [1], { trust_radius: 1, nfev: 1 }), step(1, 1, [1], { rho: 0.5 })]);
    expect(columns).toEqual(["iteration", "kind", "objective", "nfev", "trust_radius", "rho"]);
  });

  it("read a summary shaped like the service's (a stopped EnOpt run)", () => {
    const real = summary([
      { kind: "start", iteration: 0, epf_iteration: 0, x: [-1.2, 1.0], nfev: 1, njev: 0, member_evaluations: 0, objective: 281.7, step_size: 0.05 },
      { kind: "step", iteration: 1, epf_iteration: 1, x: [-1.15, 0.95], nfev: 2, njev: 1, member_evaluations: 20, objective: 233.5, step_size: 0.05 },
    ]);
    expect(objectiveChart(real.history).points.map((p) => p.mean)).toEqual([281.7, 233.5]);
    expect(historyColumns(real.history).slice(0, 7)).toEqual([
      "iteration",
      "kind",
      "objective",
      "nfev",
      "njev",
      "member_evaluations",
      "epf_iteration",
    ]);
  });

  it("show history counters as integers, and other numbers briefly", () => {
    expect(historyValue(500000, fmt)).toBe("500000");
    expect(historyValue(0.123456789, fmt)).toBe(fmt(0.123456789, 6));
    expect(historyValue(null, fmt)).toBe("–");
    expect(historyValue("step", fmt)).toBe("step");
  });

  it("say which way the objective goes from its scaling, and unknown without it", () => {
    const base = summary([]);
    expect(objectiveSense(base)).toEqual({ direction: "unknown", scaling: null });
    expect(objectiveSense({ ...base, objective_scaling: null })).toEqual({ direction: "minimised", scaling: null });
    expect(objectiveSense({ ...base, objective_scaling: 2 })).toEqual({ direction: "minimised", scaling: 2 });
    const npv = objectiveSense({ ...base, objective_scaling: -1e6 });
    expect(npv).toEqual({ direction: "maximised", scaling: -1e6 });
    // The plotted value is NPV ÷ scaling: −11.2 is an NPV of 11.2 million.
    expect(unscaled(-11.2, npv)).toBeCloseTo(11.2e6);
    expect(unscaled(3, objectiveSense({ ...base, objective_scaling: null }))).toBeNull();
    expect(plainNumber(11.2e6)).toBe("11,200,000");
    expect(plainNumber(-1e6)).toBe("-1,000,000");
    expect(plainNumber(0.0012)).toBe("0.0012");
    // The quantity beside a value: not when penalised, without a scaling, or at a scaling of 1.
    expect(objectiveQuantity(-11.2, npv, false)).toBeCloseTo(11.2e6);
    expect(objectiveQuantity(-11.2, npv, true)).toBeNull();
    expect(objectiveQuantity(undefined, npv, false)).toBeNull();
    expect(objectiveQuantity(5, objectiveSense({ ...base, objective_scaling: 1 }), false)).toBeNull();
    expect(objectiveQuantity(5, objectiveSense({ ...base, objective_scaling: null }), false)).toBeNull();
  });

  it("tell whether a constrained result meets its constraints, from the result or the latest row", () => {
    expect(feasibility(summary([step(0, 10, [1, 2])]))).toBeNull();
    // result.json as the service writes it: the optimizer's extras at the top level.
    const done = summary([step(0, 10, [1, 2], { feasible: 1.0, violation: 0 })], { x: [0.1, 0.2], objective: 3, feasible: 0.0, violation: 0.25 });
    expect(feasibility(done)).toEqual({ feasible: false, violation: 0.25, of: "result" });
    // An `extra` table is read too.
    expect(feasibility(summary([], { x: [0.1], objective: 3, extra: { feasible: 1 } }))).toEqual({ feasible: true, violation: null, of: "result" });
    // Before a result: the latest row with one.
    const running = summary([step(0, 10, [1, 2], { feasible: 1, violation: 0 }), step(1, 9, [1, 1], { feasible: 0, violation: 2 })]);
    expect(feasibility(running)).toEqual({ feasible: false, violation: 2, of: "latest" });
  });

  it("read only numbers or booleans as feasibility, never guess from other values", () => {
    expect(feasibility(summary([], { x: [1], objective: 1, feasible: null }))).toBeNull();
    expect(feasibility(summary([], { x: [1], objective: 1, feasible: "false" }))).toBeNull();
    expect(feasibility(summary([], { x: [1], objective: 1, feasible: true }))?.feasible).toBe(true);
  });

  it("name an optimization by the end of its directory", () => {
    expect(optimizationLabel("/w/UI/owner/optimizations/npv/ab12")).toBe("npv/ab12");
    expect(optimizationLabel(null)).toBe("–");
  });
});

describe("no ADR citations on any page", () => {
  it("drop them from the service's own sentences, keeping the rest", () => {
    const cases: [string, string][] = [
      ["Local analysis with GIES's RLM-MAC update in each group (ADR 0021, for GIES).", "Local analysis with GIES's RLM-MAC update in each group (for GIES)."],
      ["Field totals of a toy waterflood, for optimization tests and demos (ADR 0040/0043).", "Field totals of a toy waterflood, for optimization tests and demos."],
      ["Brier score of the ensemble's facies probabilities (Brier, 1950; ADR 0037).", "Brier score of the ensemble's facies probabilities (Brier, 1950)."],
      ["refused by the service's [policy] (ADR 0019): /w/x is outside", "refused by the service's [policy]: /w/x is outside"],
      ["[[metrics]] data_mismatch is the objective since 2026-09-30 (ADR 0027); the", "[[metrics]] data_mismatch is the objective since 2026-09-30; the"],
      ["Housekeeping of a services root (ADR 0014 section 7): reindex", "Housekeeping of a services root: reindex"],
      ["not run (ADR 0041: e.g. a localization variant)", "not run (e.g. a localization variant)"],
      ["see ADR 0039, revision 1 for the page", "for the page"],
      // Lists, several in one parenthesis, parts, "see"/"per", and a sentence's start (review).
      ["app (ADR 0015, 0017).", "app."],
      ["ADRs 0012, 0013 and 0014 cover it.", "cover it."],
      ["phases (ADR 0040, phase 1; ADR 0043) run", "phases run"],
      ["(ADR 0012; ADR 0013)", ""],
      ["guests, per ADR 0028 revisions 2 and 3, see", "guests, see"],
      ["design (ADR 0027 and its revision 1) here", "design here"],
      ["the route (see ADR 0010).", "the route."],
      ["ADR 0010. Next", "Next"],
      ["rules per ADR 0010.", "rules."],
      ["hidden (ADR 0039, rev. 4): x", "hidden: x"],
      ["as in ADR-0037, the rest", "as in the rest"],
      // Re-review: repeated prefixes, ranges, possessives, lettered phases.
      ["rules per ADR 0010 and ADR 0011 apply", "rules apply"],
      ["foo (ADR 0010 and see ADR 0011) bar", "foo bar"],
      ["records ADR 0001 to ADR 0003 hold", "records hold"],
      ["Use ADR 0010's rule", "Use the rule"],
      ["done (ADR 0038, phase B2 and C1).", "done."],
    ];
    for (const [given, shown] of cases) expect(withoutAdr(given)).toBe(shown);
  });

  it("leave everything else alone: text without citations, code, paths", () => {
    const code = "call f()  # aligned\n    x = 1 ; y = 2";
    expect(withoutAdr(code)).toBe(code);
    expect(withoutAdr("docs/adr/0024-web-ui.md")).toBe("docs/adr/0024-web-ui.md");
    expect(withoutAdr("ADRIATIC sea")).toBe("ADRIATIC sea");
    // Code beside a citation keeps its spacing.
    expect(withoutAdr("    x  =  1   (ADR 0010)")).toBe("    x  =  1");
    // Never across a line: the next line keeps its break and indentation.
    expect(withoutAdr("```\nsee ADR 0010:\n  x = 1\n```")).toBe("```\n\n  x = 1\n```");
    // Never inside a path or a name.
    for (const keep of ["/w/ADR-0037/run", "ADR0037.md", "MADR 0010", "ADR 00123", "upper"]) {
      expect(withoutAdr(keep)).toBe(keep);
    }
  });

  it("stay linear on long runs of spaces", () => {
    const long = `ADR${" ".repeat(50000)}x (${" ".repeat(30000)}ADR 0010${" ".repeat(30000)})`;
    const began = performance.now();
    withoutAdr(long);
    expect(performance.now() - began).toBeLessThan(500);
  });

  it("apply wherever server text is shown through the workspace paths", () => {
    const paths = makePaths([{ name: "", path: "/w" }]);
    expect(paths.display("refused (ADR 0019): /w/a.toml is outside")).toBe("refused: a.toml is outside");
    // Paths alone are rewritten, never filtered.
    expect(paths.path("/w/ADR-0037/run")).toBe("ADR-0037/run");
    expect(paths.display("in /w/ADR-0037/run (ADR 0010)")).toBe("in ADR-0037/run");
    // display works detached from its object too (e.g. passed to map).
    expect(["x (ADR 0010)"].map(paths.display)).toEqual(["x"]);
  });
});

describe("signing in (ADR 0024, revision 2)", () => {
  const memory = () => {
    const data = new Map<string, string>();
    return {
      getItem: (k: string) => data.get(k) ?? null,
      setItem: (k: string, v: string) => void data.set(k, v),
      removeItem: (k: string) => void data.delete(k),
      data,
    };
  };
  const stores = () => ({ local: memory(), session: memory() });
  const DAY = 24 * 60 * 60 * 1000;
  const now = Date.parse("2026-10-04T12:00:00Z");

  it("keep a sign-in for the browser session, or in this browser until its date", () => {
    const s = stores();
    saveSession(s, { token: "t1", name: "anna", until: null }, now);
    expect(s.local.data.size).toBe(0);
    expect(loadSession(s, now)).toEqual({ token: "t1", name: "anna", until: null });
    saveSession(s, { token: "t2", name: "anna", until: now + DAY }, now);
    // One place at a time: remembering forgets the session copy.
    expect(s.session.data.size).toBe(0);
    expect(loadSession(s, now)?.token).toBe("t2");
    // Past its date it is gone, and removed.
    expect(loadSession(s, now + 2 * DAY)).toBeNull();
    expect(s.local.data.size).toBe(0);
  });

  it("forget it on sign-out, and survive blocked or broken storage", () => {
    const s = stores();
    saveSession(s, { token: "t", name: "owner", until: now + DAY }, now);
    clearSession(s);
    expect(loadSession(s, now)).toBeNull();
    const blocked: Stores = { local: null, session: null };
    saveSession(blocked, { token: "t", name: "owner", until: null }, now);
    expect(loadSession(blocked, now)).toBeNull();
    const broken = stores();
    broken.local.setItem("apex.session", "{not json");
    expect(loadSession(broken, now)).toBeNull();
    const throwing = { getItem: () => { throw new Error("blocked"); }, setItem: () => { throw new Error("blocked"); }, removeItem: () => { throw new Error("blocked"); } };
    expect(loadSession({ local: throwing, session: throwing }, now)).toBeNull();
    expect(() => saveSession({ local: throwing, session: throwing }, { token: "t", name: "n", until: now + DAY }, now)).not.toThrow();
  });

  it("remember for at most the days given, and never past the token's expiry", () => {
    expect(rememberUntil(now)).toBe(now + REMEMBER_DAYS * DAY);
    expect(rememberUntil(now, "2026-10-06T12:00:00Z")).toBe(now + 2 * DAY);
    expect(rememberUntil(now, "2027-01-01T00:00:00Z")).toBe(now + REMEMBER_DAYS * DAY);
    expect(rememberUntil(now, "not a date")).toBe(now + REMEMBER_DAYS * DAY);
    expect(rememberUntil(now, null)).toBe(now + REMEMBER_DAYS * DAY);
  });

  it("accept only the token's own name: owner, or the guest's", () => {
    expect(tokenName({ role: "owner" })).toBe("owner");
    expect(nameMatches(" Owner ", { role: "owner" })).toBe(true);
    expect(nameMatches("anna", { role: "owner" })).toBe(false);
    expect(nameMatches("Anna", { role: "ui", guest: "anna" })).toBe(true);
    expect(nameMatches("bob", { role: "ui", guest: "anna" })).toBe(false);
    expect(nameMatches("owner", { role: "ui", guest: "anna" })).toBe(false);
  });

  it("tell an expired token from a refused one, and only for 401", () => {
    const problem = (status: number, title: string, type = "about:blank") =>
      new ApiError(status, { type, title, status, detail: "" });
    expect(refusal(problem(401, "Token expired"))).toBe("expired");
    expect(refusal(problem(401, "Unauthorised", "https://apex/problems/token-expired"))).toBe("expired");
    expect(refusal(problem(401, "Unauthorised"))).toBe("refused");
    expect(refusal(problem(403, "Forbidden"))).toBeNull();
    expect(refusal(new Error("network"))).toBeNull();
  });

  it("keep a remembered name and token after the user signs out, to fill in the form", () => {
    const s = stores();
    saveSession(s, { token: "t", name: "owner", until: now + DAY }, now);
    endSession(s, "t", false, now);
    // Signed out: no current sign-in, but the form can be filled in until the date.
    expect(loadSession(s, now)).toBeNull();
    expect(savedSignIn(s, now)).toEqual({ name: "owner", token: "t" });
    expect(savedSignIn(s, now + 2 * DAY)).toBeNull();
    // Signing in again makes it current; signed out again, "forget me" removes it.
    saveSession(s, { token: "t", name: "owner", until: now + DAY }, now);
    expect(loadSession(s, now)?.token).toBe("t");
    endSession(s, "t", false, now);
    forgetSaved(s, "t", now);
    expect(savedSignIn(s, now)).toBeNull();
  });

  it("forget it on an expiry or a refusal (forget), since the token is no use any more", () => {
    const s = stores();
    saveSession(s, { token: "t", name: "owner", until: now + DAY }, now);
    endSession(s, "t", true, now);
    expect(savedSignIn(s, now)).toBeNull();
    expect(loadSession(s, now)).toBeNull();
  });

  it("never end or forget another tab's sign-in (a refused old tab once wiped a new one)", () => {
    const s = stores();
    // This tab signed in for the browser session with an old token...
    saveSession(s, { token: "old", name: "owner", until: null }, now);
    // ...and another tab then signed in, remembered, with a new one (sharing localStorage).
    s.local.setItem("apex.session", JSON.stringify({ token: "new", name: "owner", until: now + DAY, active: true }));
    endSession(s, "old", true, now); // the old token is refused
    expect(loadSession(s, now)?.token).toBe("new");
    expect(JSON.parse(s.local.getItem("apex.session") ?? "{}").token).toBe("new");
  });

  it("let forget-me remove only this browser's signed-out record, never a tab's active sign-in", () => {
    const s = stores();
    saveSession(s, { token: "t", name: "owner", until: now + DAY }, now);
    // Another tab is signed in with it: forgetting from a stale form must not end that tab.
    forgetSaved(s, "t", now);
    expect(loadSession(s, now)?.token).toBe("t");
    endSession(s, "t", false, now);
    expect(savedUntil(s, "t", now)).toBe(now + DAY);
    // Not "yours" for another token.
    expect(savedUntil(s, "other", now)).toBeNull();
    forgetSaved(s, "other", now); // not the token it holds
    expect(savedSignIn(s, now)?.token).toBe("t");
    forgetSaved(s, "t", now);
    expect(savedSignIn(s, now)).toBeNull();
    expect(savedUntil(s, "t", now)).toBeNull();
  });

  it("let a sign-in for the browser session leave another tab's remembered one alone", () => {
    const s = stores();
    s.local.setItem("apex.session", JSON.stringify({ token: "a", name: "owner", until: now + DAY, active: true }));
    saveSession(s, { token: "b", name: "anna", until: null }, now);
    expect(JSON.parse(s.local.getItem("apex.session") ?? "{}").token).toBe("a");
    // This browser's own signed-out record is replaced.
    endSession(s, "a", false, now);
    saveSession(s, { token: "b", name: "anna", until: null }, now);
    expect(savedSignIn(s, now)).toBeNull();
  });

  it("put this tab's own sign-in first on reload, and pre-fill only from a signed-out record", () => {
    const s = stores();
    s.local.setItem("apex.session", JSON.stringify({ token: "a", name: "owner", until: now + DAY, active: true }));
    saveSession(s, { token: "b", name: "anna", until: null }, now);
    // Reloading this tab keeps its own sign-in, not the other tab's remembered one.
    expect(loadSession(s, now)?.token).toBe("b");
    // An active record is another tab's sign-in: nothing to pre-fill (this tab adopts it).
    expect(savedSignIn(stores(), now)).toBeNull();
    const t = stores();
    t.local.setItem("apex.session", JSON.stringify({ token: "a", name: "owner", until: now + DAY, active: true }));
    expect(savedSignIn(t, now)).toBeNull();
    endSession(t, "a", false, now);
    expect(savedSignIn(t, now)).toEqual({ name: "owner", token: "a" });
  });

  it("read records stored before sign-outs kept them as active", () => {
    const s = stores();
    s.local.setItem("apex.session", JSON.stringify({ token: "t", name: "owner", until: now + DAY }));
    expect(loadSession(s, now)?.token).toBe("t");
  });

  it("fall back to the browser session when this browser cannot keep the sign-in", () => {
    const s = stores();
    const kept = saveSession({ local: null, session: s.session }, { token: "t", name: "n", until: now + DAY }, now);
    expect(kept.until).toBeNull();
    expect(loadSession({ local: null, session: s.session }, now)).toEqual({ token: "t", name: "n", until: null });
    const ok = saveSession(s, { token: "t", name: "n", until: now + DAY }, now);
    expect(ok.until).toBe(now + DAY);
  });

  it("let only the current sign-in be ended by a refusal (a late one from an old sign-in is ignored)", () => {
    const ended: string[] = [];
    let current: string | null = "old";
    const expired = new ApiError(401, { type: "about:blank", title: "Token expired", status: 401, detail: "expired on 2026-10-11" });
    const fromOld = refusalHandler("old", () => current, (why) => ended.push(why));
    current = "new"; // signed in again while an old request was in flight
    fromOld(expired);
    expect(ended).toEqual([]);
    const fromNew = refusalHandler("new", () => current, (why) => ended.push(why));
    fromNew(new ApiError(403, { type: "about:blank", title: "Forbidden", status: 403, detail: "" }));
    expect(ended).toEqual([]);
    fromNew(expired);
    expect(ended).toHaveLength(1);
    // The service's own explanation is passed on.
    expect(ended[0]).toContain("expired on 2026-10-11");
  });

  it("read the service's own refusals (ADR 0045)", () => {
    const problem = (type: string, title: string, detail: string) =>
      new ApiError(401, { type: `https://apex/problems/${type}`, title, status: 401, detail });
    expect(refusal(problem("token-expired", "Token expired", "your web-UI token expired on 2026-10-11"))).toBe("expired");
    expect(refusal(problem("unauthorised", "Unauthorised", "send Authorization: Bearer"))).toBe("refused");
    const apiToken = problem("web-ui-token-required", "Web-UI token required", "use apex service token --ui --owner");
    expect(refusal(apiToken)).toBe("refused");
    expect(refusalText("refused", apiToken.problem.detail)).toContain("apex service token --ui --owner");
    expect(rememberUntil(Date.parse("2026-10-04T08:58:00Z"), "2026-10-07T08:58:00+00:00")).toBe(Date.parse("2026-10-07T08:58:00Z"));
  });

  it("pass on the service's explanation only where it helps a person", () => {
    const problem = (type: string, title: string, detail: string) =>
      new ApiError(401, { type: `https://apex/problems/${type}`, title, status: 401, detail });
    // An unknown token: the explanation is for programs, so none.
    expect(detailOf(problem("unauthorised", "Unauthorised", "send Authorization: Bearer <apex service token>"))).toBe("");
    expect(detailOf(new ApiError(401, { type: "about:blank", title: "Unauthorised", status: 401, detail: "send Authorization: Bearer" }))).toBe("");
    // Expired, or the API token where a web-UI token is needed: passed on.
    expect(detailOf(problem("token-expired", "Token expired", "expired on 2026-10-11"))).toBe("expired on 2026-10-11");
    expect(detailOf(problem("web-ui-token-required", "Web-UI token required", "the web UI signs in with a web-UI token"))).toContain("web-UI token");
    // An older service without types: the title still tells an expiry.
    expect(detailOf(new ApiError(401, { type: "about:blank", title: "Token expired", status: 401, detail: "d" }))).toBe("d");
    expect(detailOf(new ApiError(403, { type: "about:blank", title: "Forbidden", status: 403, detail: "d" }))).toBe("");
  });

  it("word refusals with the service's explanation when it gives one", () => {
    expect(refusalText("refused")).not.toContain("service says");
    expect(refusalText("refused", "use apex service token --ui --owner")).toContain("apex service token --ui --owner");
    expect(refusalText("expired")).toMatch(/expired/);
  });

  it("tell the app of every 401, so it signs out", async () => {
    const refused: number[] = [];
    const fetcher: Fetch = async () =>
      new Response(JSON.stringify({ title: "Token expired", status: 401, detail: "" }), { status: 401 });
    const api = new Api("t", "http://h", fetcher, (e) => refused.push(e.status));
    await expect(api.jobs()).rejects.toBeInstanceOf(ApiError);
    await expect(api.artifact("r", "a.png")).rejects.toBeInstanceOf(ApiError);
    expect(refused).toEqual([401, 401]);
  });
});

describe("the assistant's optimization drafts (ADR 0048)", () => {
  it("launch a revised run's draft as a run", () => {
    expect(launchKind({ kind: "launch_run" })).toBe("run");
    expect(launchKind({ kind: "launch_optimization" })).toBe("optimization");
    expect(launchKind({ kind: "launch_study" })).toBe("study");
    expect(launchKind({ kind: "something_new" })).toBe("study");
    const event = { n: 1, created: "", kind: "draft", data: { draft: "UI/owner/drafts/c/esmda_fixed.toml", kind: "run" } };
    expect(stepText(event)).toBe("run draft written: UI/owner/drafts/c/esmda_fixed.toml");
  });

  it("open saved reports read-only, and every configuration for editing", () => {
    expect(isReadOnly("UI/owner/drafts/c/report_esmda.md")).toBe(true);
    expect(isReadOnly("UI/owner/drafts/c/REPORT.MD")).toBe(true);
    expect(isReadOnly("5Spots/configs/gies.toml")).toBe(false);
  });

  it("launch an optimization draft as an optimization, anything else as a study", () => {
    expect(launchesOptimization({ kind: "launch_optimization" })).toBe(true);
    expect(launchesOptimization({ kind: "launch_study" })).toBe(false);
  });

  it("read the assumptions from the draft's header, only from its header", () => {
    const text = [
      "# Drafted from esmda_opm.toml by APEX's assistant (ADR 0048): a forecast from 2003-01-01 to 2006-01-01 in 6 control periods,",
      "# robust over 20 members of run esmda_opm's posterior. Review it.",
      "# Assumed: gas is not priced (wgp = 0).",
      "# Assumed: a discount rate of 8% a year.",
      "",
      "[optimization]",
      '# Assumed: not a header line any more.',
      'name = "esmda_opm_npv"',
    ].join("\n");
    expect(assumptionsOf(text)).toEqual(["gas is not priced (wgp = 0)", "a discount rate of 8% a year"]);
    expect(assumptionsOf("[optimization]\nname = 'x'")).toEqual([]);
    // A header after blank lines, and Windows line endings.
    expect(assumptionsOf("\n\n# Assumed: gas is not priced (wgp = 0).\n\n[optimization]")).toEqual(["gas is not priced (wgp = 0)"]);
    expect(assumptionsOf("# Assumed: a.\r\n# Assumed: b.\r\n\r\n[x]")).toEqual(["a", "b"]);
  });

  it("say when a draft is an optimization in the chat's steps", () => {
    const event = (data: Record<string, unknown>) => ({ n: 1, created: "", kind: "draft", data });
    expect(stepText(event({ draft: "UI/owner/drafts/c/esmda_npv.toml", kind: "optimization" }))).toBe(
      "optimization draft written: UI/owner/drafts/c/esmda_npv.toml",
    );
    expect(stepText(event({ draft: "UI/owner/drafts/c/study_x.toml" }))).toBe("draft written: UI/owner/drafts/c/study_x.toml");
  });
});

describe("a multi-objective front (ADR 0054)", () => {
  const front = {
    objective_names: ["f1", "f2"],
    x: [[0], [1], [2]],
    objectives: [
      [2, 0],
      [0, 2],
      [1, 1],
    ],
    violation: [0, 0, 0.5],
    representative: 2,
    hypervolume: 3,
    reference_point: [2.2, 2.2],
  };

  it("sorts members by the first objective and marks the representative", () => {
    const members = frontMembers(front);
    expect(members.map((m) => m.index)).toEqual([1, 2, 0]);
    expect(members.find((m) => m.representative)?.index).toBe(2);
    expect(members.find((m) => m.index === 2)?.feasible).toBe(false);
    expect(members.find((m) => m.index === 2)?.x).toEqual([2]);
  });

  it("scales parallel coordinates to the front's range; no spread is 0.5", () => {
    expect(parallelRows(front)).toEqual([
      [1, 0],
      [0, 1],
      [0.5, 0.5],
    ]);
    const flat = { ...front, objectives: [[1, 3], [1, 4]] };
    expect(parallelRows(flat)).toEqual([
      [0.5, 0],
      [0.5, 1],
    ]);
  });

  it("charts the hypervolume of the rows that have one", () => {
    const points = hypervolumePoints([
      { kind: "init", iteration: 0, hypervolume: 1 },
      { kind: "step", iteration: 1 },
      { kind: "step", iteration: 2, hypervolume: 1.5 },
    ]);
    expect(points.map((p) => [p.x, p.mean])).toEqual([
      [0, 1],
      [2, 1.5],
    ]);
  });

  it("is told apart from a single-objective run, also before its first front", () => {
    expect(isMultiObjective({ pareto: front, history: [] })).toBe(true);
    expect(isMultiObjective({ pareto: null, history: [{ kind: "init", iteration: 0, hypervolume: 0 }] })).toBe(true);
    expect(isMultiObjective({ pareto: undefined, history: [{ kind: "init", iteration: 0, objective: 1 }] })).toBe(false);
    // Early in a run (no history, no front): the service's flag or names, or objective:<name> columns.
    expect(isMultiObjective({ pareto: null, history: [], multi_objective: true })).toBe(true);
    expect(isMultiObjective({ pareto: null, history: [], objective_names: ["a", "b"] })).toBe(true);
    expect(isMultiObjective({ history: [{ kind: "step", iteration: 1, objective: 1, "objective:a": 1 }] })).toBe(true);
    expect(isMultiObjective({ pareto: null, history: [], multi_objective: false, objective_names: null })).toBe(false);
  });

  it("names the objectives maximised through a negative obj_scaling", () => {
    expect(maximisedObjectives({ objective_names: ["npv", "co2"], objective_scalings: [-1e6, null] })).toEqual(["npv"]);
    expect(maximisedObjectives({ objective_scalings: [2, -1], pareto: front })).toEqual(["f2"]);
    expect(maximisedObjectives({ objective_names: ["a"], objective_scalings: null })).toEqual([]);
    // One vector [objective] with a negative obj_scaling: every objective is maximised.
    expect(maximisedObjectives({ objective_names: ["f1", "f2"], objective_scalings: null, objective_scaling: -1 })).toEqual(["f1", "f2"]);
    expect(maximisedObjectives({ objective_scaling: -1 })).toEqual(["every objective"]);
    expect(maximisedObjectives({ objective_names: ["f1", "f2"], objective_scaling: 2 })).toEqual([]);
  });

  it("tells a front of the least infeasible member from a feasible one", () => {
    expect(frontInfeasible(front)).toBe(false);
    expect(frontInfeasible({ ...front, objectives: [[1, 1]], violation: [0.3], x: [[0]] })).toBe(true);
    expect(frontInfeasible({ ...front, objectives: [], violation: [], x: [] })).toBe(false);
  });

  it("gives each objective's best over the feasible members, not the representative's value", () => {
    // Member 2 (the representative) is infeasible: it never counts while a feasible one does.
    const lower = { ...front, objectives: [[2, 0], [0, 2], [-1, -1]] };
    expect(objectiveBests(lower)).toEqual([
      { name: "f1", best: 0, member: 1 },
      { name: "f2", best: 0, member: 0 },
    ]);
    // No feasible member: every member counts.
    const none = { ...lower, violation: [1, 1, 1] };
    expect(objectiveBests(none).map((b) => [b.best, b.member])).toEqual([
      [-1, 2],
      [-1, 2],
    ]);
    expect(objectiveBests({ ...front, objectives: [], violation: [], x: [] })).toEqual([]);
  });
});

describe("a job that reused an identical run (release tester U9)", () => {
  it("links the run when this user may open it", () => {
    expect(reusedRun({ reused: "/w/UI/owner/runs/nu", reused_visible: true, run_ids: ["abc"] })).toEqual({
      text: "Reused an identical run:",
      link: "abc",
      path: null,
    });
  });

  it("else says where it is: outside the user's web-UI folder", () => {
    const outside = reusedRun({ reused: "/w/studies/s/lg/base/r0", reused_visible: false, run_ids: ["abc"] });
    expect(outside?.link).toBeNull();
    expect(outside?.path).toBe("/w/studies/s/lg/base/r0");
    expect(outside?.text).toContain("outside your web-UI folder");
  });

  it("gives a guest no path, and says nothing for a job that ran its run", () => {
    expect(reusedRun({ reused: null, reused_visible: false, run_ids: ["abc"] })).toEqual({
      text: expect.stringContaining("outside your web-UI folder"),
      link: null,
      path: null,
    });
    expect(reusedRun({ reused: null, reused_visible: null, run_ids: ["abc"] })).toBeNull();
    expect(reusedRun({ run_ids: [] })).toBeNull(); // an older service
  });
});

describe("a submission's answer", () => {
  it("opens its own job, reused or not", () => {
    expect(submittedNote({ id: "j1", reused: false })).toEqual({ link: "j1", text: "" });
    expect(submittedNote({ id: "j1", reused: true, job_visible: true, job_state: "complete" }).link).toBe("j1");
  });

  it("never opens another submission's job the user may not see", () => {
    const elsewhere = submittedNote({ id: "j2", reused: true, job_visible: false, job_state: "running" });
    expect(elsewhere.link).toBeNull();
    expect(elsewhere.text).toContain("already queued or running outside your web-UI folder");
    expect(elsewhere.text).not.toContain("/");
    expect(submittedNote({ id: "j2", reused: true, job_visible: false, job_state: "complete" }).text).toContain(
      "exists outside your web-UI folder",
    );
  });
});

describe("the Runs pages", () => {
  it("keeps each annotation apart (release tester U6)", () => {
    expect(annotationItems({ submitted_by: "guest:bob", case_key: "7b37" })).toEqual([
      "submitted_by=guest:bob",
      "case_key=7b37",
    ]);
    expect(annotationItems({})).toEqual([]);
  });

  it("knows a missing run from other failures", () => {
    const problem = (status: number) => ({ type: "about:blank", title: "x", status, detail: "" });
    expect(isNotFound(new ApiError(404, problem(404)))).toBe(true);
    expect(isNotFound(new ApiError(403, problem(403)))).toBe(false);
    expect(isNotFound(new Error("Not found"))).toBe(false);
    expect(isNotFound(undefined)).toBe(false);
  });

  it("points to the Draw figures button when the plotter is there, else the command", () => {
    // Nothing until the agents and the user are known: no text that changes under the reader.
    expect(noFigures(false, false, "owner")).toBeNull();
    expect(noFigures(true, true, "pending")).toBeNull();
    expect(noFigures(true, true, "ui")?.text).toContain("Draw figures");
    expect(noFigures(true, true, "owner")?.command).toBeNull();
    expect(noFigures(true, false, "owner")?.command).toBe("apex report RUN_DIR --plots");
    // A guest without the plotter has no command line: ask the owner.
    expect(noFigures(true, false, "ui")).toEqual({ text: expect.stringContaining("ask the service's owner"), command: null });
    // whoami failed: not the owner, so no command line either.
    expect(noFigures(true, false, "pending", true)?.text).toContain("ask the service's owner");
    expect(noFigures(true, true, "pending", true)?.text).toContain("Draw figures");
  });
});

describe("the examples by workflow", () => {
  const example = (id: string, ...workflows: string[]) => ({
    id,
    title: id,
    summary: "",
    tags: [],
    cost: "seconds" as const,
    requires: [],
    entries: workflows.map((workflow) => ({ workflow, config: `${id}.toml` })),
  });

  it("follows the Plugins page's order, puts unknown workflows after, and drops empty ones", () => {
    const tabs = examplesByWorkflow([
      example("zdt1", "optimization"),
      example("both", "history_matching", "optimization"),
      example("new", "uncertainty_quantification"),
    ]);
    expect(tabs.map((t) => t.key)).toEqual(["history_matching", "optimization", "uncertainty_quantification"]);
    expect(tabs[1]?.examples.map((e) => e.id)).toEqual(["zdt1", "both"]);
    expect(tabs[2]?.label).toBe("Uncertainty quantification");
    expect(examplesByWorkflow([])).toEqual([]);
  });
});

describe("the assistant's Stop (ADR 0026 revision 2)", () => {
  it("posts to the chat's stop route and returns whether a turn was stopped", async () => {
    const calls: { url: string; method?: string }[] = [];
    const fetcher: Fetch = async (url, init) => {
      calls.push({ url: String(url), method: init?.method });
      return new Response(JSON.stringify({ chat: "c 1", stopped: true }), { status: 200 });
    };
    const result = await new Api("tok", "http://h", fetcher).stopChat("c 1");
    expect(calls).toEqual([{ url: "http://h/v1/chats/c%201:stop", method: "POST" }]);
    expect(result.stopped).toBe(true);
  });
});

describe("contributed web-UI pages (ADR 0064 phase 3b)", () => {
  it("links only pages served under /ui/pages/", () => {
    expect(
      contributedLinks([
        { name: "extra", title: "Extra", url: "/ui/pages/extra/" },
        { name: "away", title: "Away", url: "https://elsewhere.example/" },
        { name: "up", title: "Up", url: "/ui/pages/../api/" },
        { name: "blank", title: " ", url: "/ui/pages/blank/" },
      ]),
    ).toEqual([{ href: "/ui/pages/extra/", label: "Extra" }]);
  });
});

describe("the assistant's launch cards", () => {
  const proposal = (n: number, state: Proposal["state"]): Proposal => ({
    n, kind: "launch_study", draft: `d${n}.toml`, summary: "s", data: {}, state, job: null,
    created: "2026-10-18T00:00:00Z", updated: "2026-10-18T00:00:00Z",
  });

  it("shows only the open chat's pending cards, never another chat's (2026-10-18)", () => {
    const old = { chat: "a", list: [proposal(1, "pending"), proposal(2, "launched")] };
    expect(pendingCards(old, "a").map((p) => p.n)).toEqual([1]);
    // A new chat, created by its first message while the previous chat's cards are still held.
    expect(pendingCards(old, "b")).toEqual([]);
    expect(pendingCards(old, null)).toEqual([]);
    expect(pendingCards({ chat: null, list: [proposal(3, "pending")] }, "b")).toEqual([]);
  });
});

describe("the assistant's model chooser (ADR 0026 revision 5)", () => {
  const owner = {
    models: [
      { name: "claude", label: "Claude" },
      { name: "beam", label: "Beam" },
    ],
    default: "claude",
    choosable: true,
    note: null,
  };
  const guest = { models: [{ name: "beam", label: "Beam" }], default: "beam", choosable: false, note: "x" };

  it("keeps the owner's last pick while it is offered, else the service's default", () => {
    expect(chosenModel(owner, "beam")).toBe("beam");
    expect(chosenModel(owner, "gone")).toBe("claude");
    expect(chosenModel(owner, null)).toBe("claude");
  });

  it("never sends a model for a guest, nor for a service without a chooser", () => {
    expect(chosenModel(guest, "beam")).toBeNull();
    expect(chosenModel(null, "beam")).toBeNull();
  });

  it("keeps a reopened chat's model, and says so when the next message goes elsewhere", () => {
    // Release tester C2 (2026-10-24): a Gemma chat's follow-up was answered by Claude.
    const local = {
      ...owner,
      models: [
        { name: "claude", label: "Claude", local: false },
        { name: "gemma", label: "Gemma", local: true },
      ],
    };
    const said = (model: unknown) =>
      ({ n: 1, role: "assistant", content: "", data: model ? { chat_model: model } : {}, created: "" }) as const;
    expect(chatModel(local, [said("claude"), said("gemma"), said(null)])).toBe("gemma");
    expect(chatModel(local, [said("gone")])).toBeNull();
    expect(chatModel(local, [])).toBeNull();
    expect(switchNote(local, "gemma", "gemma")).toBeNull();
    expect(switchNote(local, null, "claude")).toBeNull(); // a new chat
    expect(switchNote(local, "gemma", "claude")).toMatch(/goes to Claude\. Claude is not on this machine/);
    expect(switchNote(local, "claude", "gemma")).not.toMatch(/leave/);
  });

  it("opens a chat with its model, a new one with the last pick, and keeps a pick made since", () => {
    expect(openingModel(owner, "beam", "claude", 0, 10)).toBe("beam"); // the chat's own
    expect(openingModel(owner, null, "beam", 0, 10)).toBe("beam"); // a new chat: the pick
    expect(openingModel(owner, "claude", "beam", 20, 10)).toBeUndefined(); // picked since
    expect(openingModel(guest, "beam", null, 0, 10)).toBeUndefined();
  });

  it("labels the model that answered", () => {
    expect(modelLabel(owner, "beam")).toBe("Beam");
    expect(modelLabel(null, "beam")).toBe("beam");
    expect(modelLabel(owner, undefined)).toBeNull();
  });

  it("sends the chosen model with a message, and none when there is none", async () => {
    const bodies: unknown[] = [];
    const fetcher: Fetch = async (_url, init) => {
      bodies.push(JSON.parse(String(init?.body)));
      return new Response(JSON.stringify({ chat: "c" }), { status: 202 });
    };
    const api = new Api("t", "", fetcher);
    await api.sendMessage("c", "hi", {}, "beam");
    await api.sendMessage("c", "hi", {}, null);
    expect(bodies).toEqual([
      { text: "hi", context: {}, model: "beam" },
      { text: "hi", context: {} },
    ]);
  });
});

describe("unmatchedOf", () => {
  it("lists the service's not-like-for-like items and ignores anything else", () => {
    expect(unmatchedOf({ data: { unmatched: ["budget: the arms' evaluations differ (a: 12, b: 29)"] } })).toEqual([
      "budget: the arms' evaluations differ (a: 12, b: 29)",
    ]);
    expect(unmatchedOf({ data: { cost: 100 } })).toEqual([]);
    expect(unmatchedOf({ data: { unmatched: ["x", 3, null] } })).toEqual(["x"]);
  });
});
