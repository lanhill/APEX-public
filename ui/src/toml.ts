// What the editor needs to know about a TOML document before the service sees it.

import { parse as parseToml } from "smol-toml";

export type Kind = "history_match" | "study" | "prediction" | "optimization";

/**
 * What a document is, as the service decides (api/app.py `_kind`): an optimization by its
 * [optimization] and [optimizer] tables, else by its first table.
 */
export function kindOf(document: Record<string, unknown>): Kind {
  if ("optimization" in document && "optimizer" in document) return "optimization";
  const first = Object.keys(document)[0];
  return first === "study" ? "study" : first === "prediction" ? "prediction" : "history_match";
}

export function dirname(path: string): string {
  const cut = path.lastIndexOf("/");
  return cut <= 0 ? "/" : path.slice(0, cut);
}

/** The TOML syntax error of `text`, if any (checked in the browser as you type). */
export function syntaxError(text: string): string | null {
  try {
    parseToml(text);
    return null;
  } catch (error) {
    return error instanceof Error ? error.message : String(error);
  }
}

/**
 * Files the workspace shows but the editor never writes: the assistant's saved reports (`.md` in
 * a drafts folder). The service accepts writes of `.toml` files only.
 */
export function isReadOnly(path: string): boolean {
  return path.toLowerCase().endsWith(".md");
}
