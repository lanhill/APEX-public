// Server-Sent Events over fetch: EventSource cannot send the bearer token.

export interface SseMessage {
  id: string | null;
  event: string;
  data: string;
}

/** Parses an SSE byte stream incrementally (https://html.spec.whatwg.org/#event-stream-interpretation). */
export class SseParser {
  private buffer = "";
  private data: string[] = [];
  private event = "";
  private id: string | null = null;

  /** Feed a chunk of text; returns the messages it completed. */
  push(chunk: string): SseMessage[] {
    this.buffer += chunk;
    const out: SseMessage[] = [];
    const lines = this.buffer.split(/\r\n|\r|\n/);
    this.buffer = lines.pop() ?? "";
    for (const line of lines) {
      if (line === "") {
        if (this.data.length) out.push({ id: this.id, event: this.event || "message", data: this.data.join("\n") });
        this.data = [];
        this.event = "";
        continue;
      }
      if (line.startsWith(":")) continue; // a comment (the service's keep-alive)
      const colon = line.indexOf(":");
      const field = colon < 0 ? line : line.slice(0, colon);
      let value = colon < 0 ? "" : line.slice(colon + 1);
      if (value.startsWith(" ")) value = value.slice(1);
      if (field === "data") this.data.push(value);
      else if (field === "event") this.event = value;
      else if (field === "id") this.id = value;
    }
    return out;
  }
}

/**
 * Follow a stream until it ends or `signal` aborts, calling `onMessage` per message.
 * Resumes from `lastEventId` (the service honours `Last-Event-ID`).
 */
export async function follow(
  url: string,
  headers: Record<string, string>,
  onMessage: (message: SseMessage) => void,
  signal: AbortSignal,
  lastEventId?: string,
): Promise<void> {
  const response = await fetch(url, {
    headers: {
      ...headers,
      Accept: "text/event-stream",
      ...(lastEventId !== undefined ? { "Last-Event-ID": lastEventId } : {}),
    },
    signal,
  });
  if (!response.ok || !response.body) throw new Error(`event stream: HTTP ${response.status}`);
  const reader = response.body.pipeThrough(new TextDecoderStream()).getReader();
  const parser = new SseParser();
  for (;;) {
    const { done, value } = await reader.read();
    if (done) return;
    for (const message of parser.push(value)) onMessage(message);
  }
}
