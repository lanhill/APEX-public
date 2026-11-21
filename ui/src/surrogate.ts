// Surrogate builds in the web UI: labels for its pages.

/** A build's label: its folder's last two parts (`<name>/<id>` where the service placed it). */
export function surrogateLabel(path: string | null): string {
  if (!path) return "–";
  const parts = path.split("/").filter(Boolean);
  return parts.slice(-2).join("/") || path;
}

/** A byte count in decimal units (`1.5 MB`). */
export function byteSize(count: number): string {
  let value = count;
  for (const unit of ["B", "kB", "MB", "GB"]) {
    if (value < 1000) return unit === "B" ? `${Math.round(value)} B` : `${Number(value.toPrecision(3))} ${unit}`;
    value /= 1000;
  }
  return `${Number(value.toPrecision(3))} TB`;
}

/** What a build's runs' state files add, an estimate; null without fields. */
export function stateFilesText(plan: {
  state_bytes_per_run?: number | null;
  state_bytes?: number | null;
  state_bytes_per_value?: number | null;
}): string | null {
  if (plan.state_bytes_per_run == null) return null;
  const total = plan.state_bytes == null ? "" : `, about ${byteSize(plan.state_bytes)} in all`;
  const per = plan.state_bytes_per_value == null ? "" : ` at about ${plan.state_bytes_per_value} bytes per value`;
  return `about ${byteSize(plan.state_bytes_per_run)} per run${total} (an estimate${per})`;
}
