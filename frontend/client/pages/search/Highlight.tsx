import type { ReactNode } from "react";
import type { TextMatchMode } from "@shared/search";

const escapeRegExp = (value: string) => value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
const TERM_RE = /[\p{L}\p{N}]+(?:['’][\p{L}\p{N}]+)*\*?/gu;

function termPattern(term: string): string {
  return term.endsWith("*")
    ? `${escapeRegExp(term.slice(0, -1))}[\\p{L}\\p{N}_]*`
    : escapeRegExp(term);
}

/**
 * Pattern for the spans a keyword matched, mirroring the backend: a phrase
 * matches its words in order across any punctuation; all/any-word modes
 * match each word. A trailing `*` matches a word prefix.
 */
export function keywordHighlightPattern(
  query: string | null | undefined,
  mode: TextMatchMode = "phrase",
): RegExp | null {
  const terms = (query ?? "").normalize("NFKC").match(TERM_RE) ?? [];
  if (terms.length === 0) return null;
  const body =
    mode === "phrase"
      ? terms.map(termPattern).join("[^\\p{L}\\p{N}_]+")
      : [...new Set(terms)].sort((a, b) => b.length - a.length).map(termPattern).join("|");
  return new RegExp(`(?<![\\p{L}\\p{N}_])(?:${body})(?![\\p{L}\\p{N}_])`, "giu");
}

/** Pattern for literal spans the API reports (`matched_terms`). */
export function literalTermsPattern(terms: readonly string[] | undefined): RegExp | null {
  const cleaned = [...new Set((terms ?? []).map((term) => term.trim()).filter(Boolean))];
  if (cleaned.length === 0) return null;
  return new RegExp(
    cleaned.sort((a, b) => b.length - a.length).map(escapeRegExp).join("|"),
    "giu",
  );
}

/** Renders `text` with every match of `pattern` wrapped in <mark>. */
export function Highlight({ text, pattern }: { text: string; pattern: RegExp | null }) {
  if (!pattern || !text) return <>{text}</>;
  const parts: ReactNode[] = [];
  let last = 0;
  // A fresh RegExp per render: `g` patterns are stateful.
  const re = new RegExp(pattern.source, pattern.flags.includes("g") ? pattern.flags : `${pattern.flags}g`);
  for (const match of text.matchAll(re)) {
    const start = match.index ?? 0;
    if (match[0].length === 0) continue;
    if (start > last) parts.push(text.slice(last, start));
    parts.push(
      <mark
        key={start}
        // Same treatment as the reader's find highlight, so a hit reads the
        // same in results and in the agreement.
        className="rounded-sm bg-warning/25 px-0.5 text-foreground ring-1 ring-warning/40"
      >
        {match[0]}
      </mark>,
    );
    last = start + match[0].length;
  }
  if (last < text.length) parts.push(text.slice(last));
  return <>{parts}</>;
}
