// Hash routes (#/jobs/<id>): the service serves one index.html for every page.

import { useEffect, useState } from "react";

export type Route =
  | { page: "jobs" }
  | { page: "job"; id: string }
  | { page: "runs"; filter?: string }
  | { page: "run"; id: string }
  | { page: "studies" }
  | { page: "study"; name: string }
  | { page: "prediction"; path: string }
  | { page: "optimizations" }
  | { page: "optimization"; path: string }
  | { page: "plugins"; workflow?: string; registry?: string; name?: string }
  | { page: "examples"; workflow?: string }
  | { page: "agents"; name?: string }
  | { page: "editor"; path?: string };

export function parse(hash: string): Route {
  const [path = "", search = ""] = hash.replace(/^#\/?/, "").split("?");
  const parts = path.split("/").filter(Boolean).map(decodeURIComponent);
  const params = new URLSearchParams(search);
  const [head, id, second, third] = parts;
  switch (head) {
    case "jobs":
      return id ? { page: "job", id } : { page: "jobs" };
    case "runs":
      if (id) return { page: "run", id };
      return params.get("filter") ? { page: "runs", filter: params.get("filter") as string } : { page: "runs" };
    case "studies":
      return id ? { page: "study", name: id } : { page: "studies" };
    case "prediction":
      return { page: "prediction", path: params.get("path") ?? "" };
    case "optimizations":
      // #/optimizations: the list; #/optimizations?path=DIR: one optimization.
      return params.get("path") ? { page: "optimization", path: params.get("path") as string } : { page: "optimizations" };
    case "plugins":
      // #/plugins/<workflow>/<registry>/<name>: a workflow's tab, and one plugin's details.
      return {
        page: "plugins",
        ...(id ? { workflow: id } : {}),
        ...(id && second && third ? { registry: second, name: third } : {}),
      };
    case "agents":
      return id ? { page: "agents", name: id } : { page: "agents" };
    case "examples":
      // #/examples/<workflow>: the catalogue's tab of one workflow.
      return id ? { page: "examples", workflow: id } : { page: "examples" };
    case "editor":
      return params.get("path") ? { page: "editor", path: params.get("path") as string } : { page: "editor" };
    default:
      return { page: "jobs" };
  }
}

export function href(route: Route): string {
  switch (route.page) {
    case "job":
      return `#/jobs/${encodeURIComponent(route.id)}`;
    case "run":
      return `#/runs/${encodeURIComponent(route.id)}`;
    case "study":
      return `#/studies/${encodeURIComponent(route.name)}`;
    case "prediction":
      return `#/prediction?path=${encodeURIComponent(route.path)}`;
    case "optimization":
      return `#/optimizations?path=${encodeURIComponent(route.path)}`;
    case "runs":
      return route.filter ? `#/runs?filter=${encodeURIComponent(route.filter)}` : "#/runs";
    case "editor":
      return route.path ? `#/editor?path=${encodeURIComponent(route.path)}` : "#/editor";
    case "agents":
      return route.name ? `#/agents/${encodeURIComponent(route.name)}` : "#/agents";
    case "examples":
      return route.workflow ? `#/examples/${encodeURIComponent(route.workflow)}` : "#/examples";
    case "plugins": {
      const parts = [route.workflow, ...(route.workflow && route.registry && route.name ? [route.registry, route.name] : [])];
      return ["#/plugins", ...parts.filter((x): x is string => !!x).map(encodeURIComponent)].join("/");
    }
    default:
      return `#/${route.page}`;
  }
}

export function useRoute(): Route {
  const [route, setRoute] = useState(() => parse(window.location.hash));
  useEffect(() => {
    const onChange = () => setRoute(parse(window.location.hash));
    window.addEventListener("hashchange", onChange);
    return () => window.removeEventListener("hashchange", onChange);
  }, []);
  return route;
}
