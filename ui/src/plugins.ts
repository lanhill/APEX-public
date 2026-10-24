// The Plugins page's model: plugins grouped by the workflow they serve, and their options read
// from the JSON schema the API sends. Display only: every fact comes from the API.

import type { PluginInfo } from "./api";

export interface Workflow {
  /** In the page's address, e.g. `#/plugins/history_matching`. */
  key: string;
  label: string;
  /** Its registries, in the order the page lists them. */
  registries: string[];
}

/**
 * The workflows, in the page's order, and which registries each serves. A service that sends
 * each plugin's `workflow` decides it; for older ones, these registries are the fallback.
 * Simulators and executors run members for every workflow, so they have a tab of their own.
 * A plugin of no workflow (or of a registry not named here) is shown under "Other".
 */
export const WORKFLOWS: Workflow[] = [
  {
    key: "history_matching",
    label: "History matching",
    registries: [
      "methods",
      "update_strategies",
      "localizations",
      "damping_schemes",
      "priors",
      "parameterizations",
      "data_transforms",
      "metrics",
      "stopping_rules",
    ],
  },
  { key: "prediction", label: "Prediction", registries: ["forecast_scores"] },
  {
    key: "optimization",
    label: "Optimization",
    registries: ["optimizers", "objectives", "response_objectives", "step_rules"],
  },
  { key: "simulation", label: "Simulation & execution", registries: ["simulators", "executors"] },
  { key: "reporting", label: "Figures", registries: ["figures", "figure_presets"] },
];

export const OTHER: Workflow = { key: "other", label: "Other", registries: [] };

/** A registry's name as a heading: `update_strategies` → "Update strategies". */
export function registryLabel(registry: string): string {
  const words = registry.replace(/_/g, " ");
  return words.charAt(0).toUpperCase() + words.slice(1);
}

export interface RegistryGroup {
  registry: string;
  plugins: PluginInfo[];
}

export interface WorkflowGroup {
  workflow: Workflow;
  registries: RegistryGroup[];
  count: number;
}

/**
 * The plugins by workflow and registry, in the order of `WORKFLOWS` (plugins in the API's
 * order). A plugin's workflow is the one it declares, else its registry's in `WORKFLOWS`; none,
 * or one the page does not know, puts it under "Other". Workflows without plugins are left out.
 */
export function groupPlugins(plugins: PluginInfo[]): WorkflowGroup[] {
  const fallback = new Map(WORKFLOWS.flatMap((w) => w.registries.map((r) => [r, w.key] as const)));
  const known = new Set(WORKFLOWS.map((w) => w.key));
  const workflowOf = (p: PluginInfo): string => {
    const declared = p.workflow === undefined ? fallback.get(p.registry) : p.workflow;
    return declared && known.has(declared) ? declared : OTHER.key;
  };
  // workflow -> registry -> plugins, in the API's order.
  const byWorkflow = new Map<string, Map<string, PluginInfo[]>>();
  for (const p of plugins) {
    const workflow = workflowOf(p);
    const registries = byWorkflow.get(workflow) ?? new Map<string, PluginInfo[]>();
    registries.set(p.registry, [...(registries.get(p.registry) ?? []), p]);
    byWorkflow.set(workflow, registries);
  }
  return [...WORKFLOWS, OTHER]
    .map((workflow) => {
      const found = byWorkflow.get(workflow.key) ?? new Map<string, PluginInfo[]>();
      // The known registries in the page's order, then any others in the API's.
      const named = workflow.registries.filter((r) => found.has(r));
      const others = [...found.keys()].filter((r) => !workflow.registries.includes(r));
      const order = [...named, ...others];
      const registries = order.map((registry) => ({ registry, plugins: found.get(registry) ?? [] }));
      return {
        workflow: { ...workflow, registries: order },
        registries,
        count: registries.reduce((n, r) => n + r.plugins.length, 0),
      };
    })
    .filter((g) => g.count > 0);
}

/** The plugins of a group whose name or summary contains `text` (any case); empty registries dropped. */
export function filterGroup(group: WorkflowGroup, text: string): RegistryGroup[] {
  const wanted = text.trim().toLowerCase();
  if (!wanted) return group.registries;
  return group.registries
    .map((r) => ({
      registry: r.registry,
      plugins: r.plugins.filter(
        (p) => p.name.toLowerCase().includes(wanted) || p.summary.toLowerCase().includes(wanted),
      ),
    }))
    .filter((r) => r.plugins.length > 0);
}

type Schema = Record<string, unknown>;

export interface OptionRow {
  name: string;
  type: string;
  required: boolean;
  /** The default as TOML-like text; undefined when there is none. */
  default?: string;
  /** Allowed values, for an enum. */
  choices?: string[];
  description?: string;
}

/** A plugin's options, one row each, from its JSON schema (pydantic's). */
export function optionRows(options: Record<string, unknown>): OptionRow[] {
  const properties = (options.properties ?? {}) as Record<string, Schema>;
  const required = new Set((options.required as string[] | undefined) ?? []);
  const defs = (options.$defs ?? {}) as Record<string, Schema>;
  return Object.entries(properties).map(([name, schema]) => {
    const resolved = resolve(schema, defs);
    const choices = enumOf(resolved, defs);
    return {
      name,
      type: typeText(resolved, defs),
      required: required.has(name),
      ...("default" in schema ? { default: valueText(schema.default) } : {}),
      ...(choices ? { choices } : {}),
      ...(typeof schema.description === "string" ? { description: schema.description } : {}),
    };
  });
}

function resolve(schema: Schema, defs: Record<string, Schema>): Schema {
  const ref = schema.$ref;
  if (typeof ref === "string" && ref.startsWith("#/$defs/")) {
    return defs[ref.slice("#/$defs/".length)] ?? schema;
  }
  return schema;
}

function enumOf(schema: Schema, defs: Record<string, Schema>): string[] | undefined {
  if (Array.isArray(schema.enum)) return schema.enum.map(valueText);
  for (const option of (schema.anyOf as Schema[] | undefined) ?? []) {
    const found = enumOf(resolve(option, defs), defs);
    if (found) return found;
  }
  return undefined;
}

/** A schema's type, briefly: `number`, `list of string`, `string or null`, `table`. */
export function typeText(schema: Schema, defs: Record<string, Schema> = {}): string {
  const s = resolve(schema, defs);
  if (Array.isArray(s.anyOf)) {
    const parts = (s.anyOf as Schema[]).map((x) => typeText(x, defs));
    return [...new Set(parts)].join(" or ");
  }
  if (Array.isArray(s.enum)) return "choice";
  if (s.format === "path") return "path";
  switch (s.type) {
    case "array":
      return s.items ? `list of ${typeText(s.items as Schema, defs)}` : "list";
    case "object":
      return "table";
    case "integer":
    case "number":
    case "string":
    case "boolean":
    case "null":
      return s.type;
    default:
      return typeof s.title === "string" && s.title ? s.title : "any";
  }
}

/** A value as it would be written in a configuration: strings quoted, null as `none`. */
export function valueText(value: unknown): string {
  if (value === null || value === undefined) return "none";
  if (typeof value === "string") return JSON.stringify(value);
  return JSON.stringify(value);
}

/** Summaries are reStructuredText: ``code`` becomes `code` (rendered as code by the page). */
export function summaryParts(summary: string): { text: string; code: boolean }[] {
  return summary
    .split(/``([^`]+)``/)
    .map((text, i) => ({ text, code: i % 2 === 1 }))
    .filter((p) => p.text !== "");
}
