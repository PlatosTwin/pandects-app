import type { SearchFilters } from "./sections";

export type SearchMode = "sections" | "transactions" | "tax";

export function parseSearchMode(value: string | null | undefined): SearchMode {
  if (value === "transactions") return "transactions";
  if (value === "tax") return "tax";
  return "sections";
}

export type SharedSearchFilters = SearchFilters;
