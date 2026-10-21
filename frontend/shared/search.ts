import type { SearchFilters } from "./sections";

export type SearchMode = "sections" | "transactions" | "tax";

export function parseSearchMode(value: string | null | undefined): SearchMode {
  if (value === "transactions") return "transactions";
  if (value === "tax") return "tax";
  return "sections";
}

/** Keyword matching for `text_query` (backend `text_match_mode`). */
export type TextMatchMode = "phrase" | "all_terms" | "any_terms";

export function parseTextMatchMode(value: string | null | undefined): TextMatchMode {
  if (value === "all_terms") return "all_terms";
  if (value === "any_terms") return "any_terms";
  return "phrase";
}

/**
 * Filters for any search mode. `include_rep_warranty` only affects tax-clause
 * searches; the other modes ignore it. `text_query` / `text_match_mode` are the
 * keyword search (the URL carries them as `q` / `match`); only modes whose
 * config sends them (see KEYWORD_SEARCH_SUPPORT) use them.
 */
export type SearchQueryFilters = SearchFilters & {
  include_rep_warranty?: boolean;
  text_query?: string;
  text_match_mode?: TextMatchMode;
};
