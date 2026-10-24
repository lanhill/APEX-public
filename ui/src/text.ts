// Server text as the web UI shows it. The service's messages, plugin summaries and agents'
// reports cite APEX's design records ("ADR 0037"); the owner asked (2026-10-04) that no page
// show them, so the citation goes and the rest of the sentence stays.

/**
 * A citation's own part: "section 7", "revision 1", "revisions 2 and 3", "rev. 4", "phase 1",
 * "phase B2 and C1". Further items are numbers or letter codes only, so a following word
 * (", see") is never taken for one.
 */
const PART = String.raw`(?:and[ \t]+its[ \t]+)?(?:sections?|revisions?|rev\.?|phases?)[ \t]+\w+(?:\.\d+)?(?:[ \t]{0,4}(?:,|and|&)[ \t]{0,4}(?:\d+|[A-Z]\d*)\b)*`;

/**
 * One citation, with a "see", "per" or "cf." before it: "ADR 0037", "ADR-0037", "ADRs 0012,
 * 0013 and 0014", "ADR 0040/0043", "ADR 0014 section 7", "ADR 0039, revision 1". Whitespace is
 * spaces and tabs only, and bounded, so a citation never spans lines and matching stays linear.
 * Lists may repeat the prefix ("ADR 0010 and ADR 0011", "ADR 0001 to ADR 0003"). It stands
 * alone: nothing of a path or name before it ("/w/ADR-0037/run", "MADR 0010") or after it
 * ("ADR0037.md").
 */
const CITATION = new RegExp(
  String.raw`(?<![\w/.-])(?:(?:see|per|cf\.)[ \t]{1,4})?ADRs?[ -]?\d{4}\b(?:[ \t]{0,4}(?:,|/|and|&|to)[ \t]{0,4}(?:(?:see|per|cf\.)[ \t]{1,4})?(?:ADRs?[ -]?)?\d{4}\b)*(?:,?[ \t]{1,4}${PART})*(?![\w-]|[./][A-Za-z_])`,
  "g",
);

/**
 * Where a citation was, while the punctuation around it is tidied. Text that already holds
 * this character is shown as it is, unfiltered, rather than risk mangling it.
 */
const MARK = "\u0000";

/**
 * `text` without its ADR citations, the rest of the sentence kept: a parenthesis holding only
 * citations goes (with the space before it), and a citation goes with its separator:
 * "(ADR 0021, for GIES)" → "(for GIES)", "(Brier, 1950; ADR 0037)" → "(Brier, 1950)",
 * "refused (ADR 0019): x" → "refused: x", "ADR 0010. Next" → "Next". Line breaks and every
 * other character stay as they were (code blocks keep their layout).
 */
export function withoutAdr(text: string): string {
  if (!text.includes("ADR") || text.includes(MARK)) return text;
  // A possessive citation stands for its record: "ADR 0010's rule" → "the rule".
  const marked = text.replace(CITATION, MARK).replace(/\u0000['’]s\b/g, "the");
  if (marked === text) return text;
  return marked
    .replace(/[ \t]{0,8}\([ \t]{0,4}\u0000(?:[ \t]{0,4}[,;][ \t]{0,4}\u0000)*[ \t]{0,4}\)/g, "")
    .replace(/^([ \t]*)\u0000[ \t]{0,4}[.,;:]?[ \t]{0,4}/gm, "$1")
    .replace(/\u0000[ \t]{0,4}[,;:][ \t]{0,4}/g, "")
    .replace(/[ \t]{0,8}[,;][ \t]{0,4}\u0000/g, "")
    .replace(/[ \t]{0,8}\u0000/g, "");
}
