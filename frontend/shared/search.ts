import type { SearchFilters } from "./sections";

export type SearchMode = "sections" | "transactions" | "tax";

export function parseSearchMode(value: string | null | undefined): SearchMode {
  if (value === "transactions") return "transactions";
  if (value === "tax") return "tax";
  return "sections";
}

/**
 * Filters for any search mode. `include_rep_warranty` only affects tax-clause
 * searches; the other modes ignore it.
 */
export type SearchQueryFilters = SearchFilters & {
  include_rep_warranty?: boolean;
};
