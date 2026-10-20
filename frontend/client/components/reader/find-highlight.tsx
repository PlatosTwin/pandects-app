import type { ReactNode } from "react";

/** DOM contract between the highlighter and the find bar. */
export const FIND_HIT_ATTR = "data-find-hit";
export const FIND_CURRENT_ATTR = "data-find-current";

const REGEX_SPECIAL_RE = /[.*+?^${}()|[\]\\]/g;

/**
 * Case-insensitive matcher for a find query. Runs of whitespace in the query
 * match any whitespace run in the text, so "material adverse" still matches
 * across a line break. Returns null for a blank query.
 */
export function buildFindRegex(query: string): RegExp | null {
  const trimmed = query.trim();
  if (!trimmed) return null;
  const pattern = trimmed
    .split(/\s+/)
    .map((part) => part.replace(REGEX_SPECIAL_RE, "\\$&"))
    .join("\\s+");
  return new RegExp(pattern, "gi");
}

export interface TextSegment {
  text: string;
  hit: boolean;
}

/** Splits `text` into alternating plain and matching segments. */
export function splitFindSegments(text: string, regex: RegExp | null): TextSegment[] {
  if (!regex || !text) return [{ text, hit: false }];
  const segments: TextSegment[] = [];
  let cursor = 0;
  regex.lastIndex = 0;
  for (const match of text.matchAll(regex)) {
    const start = match.index ?? 0;
    const value = match[0];
    if (!value) continue;
    if (start > cursor) segments.push({ text: text.slice(cursor, start), hit: false });
    segments.push({ text: value, hit: true });
    cursor = start + value.length;
  }
  if (cursor < text.length) segments.push({ text: text.slice(cursor), hit: false });
  return segments.length > 0 ? segments : [{ text, hit: false }];
}

export const findHitClassName =
  "rounded-sm bg-warning/20 px-px text-foreground ring-1 ring-warning/30 data-[find-current]:bg-warning data-[find-current]:text-warning-foreground data-[find-current]:ring-warning";

/** Renders `text` with every hit wrapped in `<mark data-find-hit>`. */
export function renderFindHighlights(text: string, regex: RegExp | null): ReactNode {
  if (!regex) return text;
  const segments = splitFindSegments(text, regex);
  if (segments.length === 1 && !segments[0].hit) return text;
  return segments.map((segment, index) =>
    segment.hit ? (
      <mark key={index} {...{ [FIND_HIT_ATTR]: "" }} className={findHitClassName}>
        {segment.text}
      </mark>
    ) : (
      segment.text
    ),
  );
}
