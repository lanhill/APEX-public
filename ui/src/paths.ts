// Workspace-relative paths. The browser shows paths relative to the
// workspace, and sends them so too: every route takes `<root name>/<rest>` (with one root,
// the path under it), so the browser needs no server path (finding 31 of the release
// tester, 2026-10-19). A guest's workspace has names only; the owner's also has each root's
// folder, which `path` and `display` rewrite in the server's texts.

import { withoutAdr } from "./text";

export interface Root {
  name: string; // "" when there is one root: paths are relative to it
  /** The root's folder on the server: the owner's only (absent for guests). */
  path?: string;
}

export interface Paths {
  /** The path a route takes for a workspace-relative path: itself, checked and normalised (an absolute one as it is). */
  toServer(relative: string): string;
  /** The path a route takes for the folder of a workspace-relative file (`.`: the only root). */
  folderOf(relative: string): string;
  /** `path` (or text of paths) with every server path under a root relative to the workspace. */
  path(text: string): string;
  /** Text as a page shows it: its paths as `path` shows them, and no ADR citations (`withoutAdr`). */
  display(text: string): string;
}

export function makePaths(roots: Root[]): Paths {
  // Longest first, so that a root inside another is matched before it.
  const known = roots.filter((r): r is Required<Root> => typeof r.path === "string" && r.path !== "");
  const sorted = [...known].sort((a, b) => b.path.length - a.path.length);
  const path = (text: string): string => {
    let out = text;
    for (const root of sorted) {
      const prefix = root.path.endsWith("/") ? root.path : `${root.path}/`;
      out = out.split(prefix).join(root.name ? `${root.name}/` : "");
      // The root itself, not followed by a path.
      out = out.replace(new RegExp(`${escape(root.path)}(?![\\w./-])`, "g"), root.name || "(workspace)");
    }
    return out;
  };
  const toServer = (relative: string): string => {
    // An absolute path (the owner's, typed or shown in full) goes as it is: the server takes it.
    if (relative.startsWith("/")) return relative.length > 1 ? relative.replace(/\/+$/, "") : relative;
    const clean = relative.replace(/\/+$/, "");
    if (roots.length === 1) return clean || ".";
    const [head = ""] = clean.split("/");
    if (!roots.some((r) => r.name === head)) throw new Error(`${relative}: not under a workspace root`);
    return clean;
  };
  return {
    toServer,
    folderOf(relative) {
      const cut = relative.lastIndexOf("/");
      if (relative.startsWith("/")) return cut <= 0 ? "/" : relative.slice(0, cut);
      return toServer(cut < 0 ? "" : relative.slice(0, cut));
    },
    path,
    // Paths first, then citations; a citation stands alone, so a path is never taken for one.
    display: (text) => withoutAdr(path(text)),
  };
}

function escape(text: string): string {
  return text.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

/** Paths before the workspace is known: nothing is rewritten. */
export const NO_PATHS: Paths = {
  toServer: (relative) => relative,
  folderOf: (relative) => {
    const cut = relative.lastIndexOf("/");
    return cut <= 0 ? "." : relative.slice(0, cut);
  },
  path: (text) => text,
  display: (text) => withoutAdr(text),
};
