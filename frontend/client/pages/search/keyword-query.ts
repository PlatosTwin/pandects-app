/**
 * Client-side mirror of the backend keyword rules (backend/text_search.py and
 * the `text_query` validators in backend/schemas/sections.py). The API answers
 * a rejected keyword with a generic 422, so the query bar checks these first
 * and explains the problem inline.
 */
import type { TextMatchMode } from "@shared/search";

export const MAX_KEYWORD_CHARS = 256;
const MAX_TERMS = 24;
const MAX_ANY_TERMS = 8;
const MAX_PREFIX_TERMS = 4;
const MIN_PREFIX_CHARS = 3;

// MariaDB's default InnoDB FULLTEXT stopwords (backend DEFAULT_INNODB_STOPWORDS).
const STOPWORDS = new Set([
  "a", "about", "an", "are", "as", "at", "be", "by", "com", "de", "en", "for",
  "from", "how", "i", "in", "is", "it", "la", "of", "on", "or", "that", "the",
  "this", "to", "und", "was", "what", "when", "where", "who", "will", "with", "www",
]);

const TERM_RE = /[\p{L}\p{N}]+(?:['’][\p{L}\p{N}]+)*\*?/gu;
const ALNUM_RE = /[\p{L}\p{N}]/u;

export const TEXT_MATCH_MODE_LABELS: Record<TextMatchMode, string> = {
  phrase: "Exact phrase",
  all_terms: "All words",
  any_terms: "Any word",
};

function isIndexedTerm(term: string): boolean {
  return term.endsWith("*") || (term.length >= MIN_PREFIX_CHARS && !STOPWORDS.has(term));
}

/** Returns a user-facing problem with the keyword, or null when it can be searched. */
export function validateKeywordQuery(raw: string, mode: TextMatchMode): string | null {
  const normalized = raw.normalize("NFKC").toLowerCase().split(/\s+/).filter(Boolean).join(" ");
  if (!normalized) return "Enter a word or phrase to search for.";
  if (normalized.length > MAX_KEYWORD_CHARS) {
    return `Keep the keyword under ${MAX_KEYWORD_CHARS} characters.`;
  }
  for (let i = 0; i < normalized.length; i += 1) {
    if (normalized[i] !== "*") continue;
    const previous = i > 0 ? normalized[i - 1] : "";
    const following = normalized[i + 1] ?? "";
    if (
      !ALNUM_RE.test(previous) ||
      (following && (ALNUM_RE.test(following) || "_'’*".includes(following)))
    ) {
      return "Use * only at the end of a word, for example indemnif*.";
    }
  }
  const terms = normalized.match(TERM_RE) ?? [];
  if (terms.length === 0) return "Include at least one letter or number.";
  if (terms.length > MAX_TERMS) return `Use at most ${MAX_TERMS} words.`;
  const prefixTerms = terms.filter((term) => term.endsWith("*"));
  if (prefixTerms.length > MAX_PREFIX_TERMS) {
    return `Use at most ${MAX_PREFIX_TERMS} words ending in *.`;
  }
  if (prefixTerms.some((term) => term.length - 1 < MIN_PREFIX_CHARS)) {
    return `Put at least ${MIN_PREFIX_CHARS} letters before *.`;
  }
  if (mode === "phrase") {
    if (!terms.some(isIndexedTerm)) {
      return "Add a longer word: short, common words such as “the” or “of” can't be searched on their own.";
    }
    return null;
  }
  if (mode === "any_terms" && terms.length > MAX_ANY_TERMS) {
    return `“Any word” searches can use at most ${MAX_ANY_TERMS} words.`;
  }
  if (!terms.every(isIndexedTerm)) {
    return "“All words” and “Any word” can't match short, common words such as “the” or “of”. Remove them or use “Exact phrase”.";
  }
  return null;
}
