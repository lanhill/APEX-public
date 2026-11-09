// Surrogate builds in the web UI: labels for its pages.

/** A build's label: its folder's last two parts (`<name>/<id>` where the service placed it). */
export function surrogateLabel(path: string | null): string {
  if (!path) return "–";
  const parts = path.split("/").filter(Boolean);
  return parts.slice(-2).join("/") || path;
}
