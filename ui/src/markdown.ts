// A small Markdown reader for the assistant's replies and saved reports: paragraphs, headings,
// bullet and numbered lists, code blocks, tables (`| a | b |` rows), `code`, **bold** and *italic*. It builds a tree that React
// renders as elements, never as HTML, so a reply cannot inject markup.

export type Inline = { kind: "text" | "code" | "bold" | "italic"; text: string };
export type Block =
  | { kind: "heading"; level: number; inline: Inline[] }
  | { kind: "paragraph"; inline: Inline[] }
  | { kind: "list"; ordered: boolean; items: Inline[][] }
  | { kind: "code"; text: string }
  | { kind: "table"; header: Inline[][]; rows: Inline[][][] };

/** A table row's cells: `| a | b |` → ["a", "b"] (the outer pipes are optional). */
function cells(line: string): string[] {
  const inner = line.trim().replace(/^\|/, "").replace(/\|$/, "");
  return inner.split("|").map((c) => c.trim());
}

/** Whether a line is a table's separator row: `|---|:---:|`. */
function separator(line: string): boolean {
  return /^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)*\|?\s*$/.test(line);
}

export function inline(text: string): Inline[] {
  const out: Inline[] = [];
  const pattern = /(`[^`]+`|\*\*[^*]+\*\*|\*[^*\s][^*]*\*)/g;
  let last = 0;
  for (const match of text.matchAll(pattern)) {
    const at = match.index ?? 0;
    if (at > last) out.push({ kind: "text", text: text.slice(last, at) });
    const token = match[0];
    if (token.startsWith("`")) out.push({ kind: "code", text: token.slice(1, -1) });
    else if (token.startsWith("**")) out.push({ kind: "bold", text: token.slice(2, -2) });
    else out.push({ kind: "italic", text: token.slice(1, -1) });
    last = at + token.length;
  }
  if (last < text.length) out.push({ kind: "text", text: text.slice(last) });
  return out;
}

export function parse(markdown: string): Block[] {
  const blocks: Block[] = [];
  const lines = markdown.replace(/\r\n/g, "\n").split("\n");
  let paragraph: string[] = [];
  const flush = () => {
    if (paragraph.length) blocks.push({ kind: "paragraph", inline: inline(paragraph.join(" ")) });
    paragraph = [];
  };
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i] ?? "";
    if (line.startsWith("```")) {
      flush();
      const code: string[] = [];
      for (i++; i < lines.length && !(lines[i] ?? "").startsWith("```"); i++) code.push(lines[i] ?? "");
      blocks.push({ kind: "code", text: code.join("\n") });
      continue;
    }
    // A table: a row of cells, its separator row, then rows (reports' "Key numbers").
    // The separator holds a pipe and as many cells as the header, so a lone "| a" before a
    // "---" rule (a horizontal rule, no pipe) is not a table.
    const next = lines[i + 1] ?? "";
    if (line.trim().startsWith("|") && next.includes("|") && separator(next) && cells(next).length === cells(line).length) {
      flush();
      const header = cells(line).map(inline);
      const rows: Inline[][][] = [];
      for (i += 2; i < lines.length && (lines[i] ?? "").trim().startsWith("|"); i++) {
        rows.push(cells(lines[i] ?? "").map(inline));
      }
      i--;
      blocks.push({ kind: "table", header, rows });
      continue;
    }
    const heading = /^(#{1,4})\s+(.*)$/.exec(line);
    if (heading) {
      flush();
      blocks.push({ kind: "heading", level: heading[1]?.length ?? 1, inline: inline(heading[2] ?? "") });
      continue;
    }
    const item = /^\s*(?:([-*])|(\d+)[.)])\s+(.*)$/.exec(line);
    if (item) {
      flush();
      const ordered = item[2] !== undefined;
      const previous = blocks[blocks.length - 1];
      const content = inline(item[3] ?? "");
      if (previous && previous.kind === "list" && previous.ordered === ordered) previous.items.push(content);
      else blocks.push({ kind: "list", ordered, items: [content] });
      continue;
    }
    if (!line.trim()) {
      flush();
      continue;
    }
    paragraph.push(line.trim());
  }
  flush();
  return blocks;
}
