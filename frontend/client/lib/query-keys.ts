import type { FilterOptionsResponse } from "@shared/sections";

/**
 * Centralized React Query key factory.
 *
 * Keep all query keys in this file. The shape `[scope, ...args]` allows
 * `queryClient.invalidateQueries({ queryKey: keys.sections.all })` to nuke
 * everything under a scope without listing each variant.
 */

function sortedFields(fields?: ReadonlyArray<keyof FilterOptionsResponse>) {
  if (!fields || fields.length === 0) return null;
  return Array.from(new Set(fields)).sort();
}

export const keys = {
  filterOptions: {
    all: ["filter-options"] as const,
    list: (fields?: ReadonlyArray<keyof FilterOptionsResponse>) =>
      ["filter-options", sortedFields(fields)] as const,
  },
  taxonomy: {
    all: ["taxonomy"] as const,
  },
  taxClauseTaxonomy: {
    all: ["tax-clause-taxonomy"] as const,
  },
  agreement: {
    all: ["agreement"] as const,
    detail: (uuid: string) => ["agreement", uuid] as const,
    summary: (uuid: string) => ["agreement", uuid, "summary"] as const,
  },
  sections: {
    all: ["sections"] as const,
    search: (params: Record<string, unknown>) =>
      ["sections", "search", params] as const,
  },
  taxClauses: {
    all: ["tax-clauses"] as const,
    search: (params: Record<string, unknown>) =>
      ["tax-clauses", "search", params] as const,
  },
  transactions: {
    all: ["transactions"] as const,
    search: (params: Record<string, unknown>) =>
      ["transactions", "search", params] as const,
  },
  favorites: {
    all: ["favorites"] as const,
    list: ["favorites", "list"] as const,
    tags: ["favorites", "tags"] as const,
    projects: ["favorites", "projects"] as const,
  },
} as const;
