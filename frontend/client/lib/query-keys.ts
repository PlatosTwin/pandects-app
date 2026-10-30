import type { FilterOptionsResponse } from "@shared/sections";

/**
 * Centralized React Query key factory.
 *
 * Every query key lives in this file. Keys are `[scope, ...args]`, so a
 * prefix such as `keys.favorites.all` matches every query under that scope
 * (`queryClient.removeQueries({ queryKey: keys.favorites.all })`).
 */

function sortedFields(fields?: ReadonlyArray<keyof FilterOptionsResponse>) {
  if (!fields || fields.length === 0) return null;
  return Array.from(new Set(fields)).sort();
}

export type TaxonomyKind = "main" | "tax";

export const keys = {
  filterOptions: {
    list: (fields?: ReadonlyArray<keyof FilterOptionsResponse>) =>
      ["filter-options", sortedFields(fields)] as const,
    /** Type-ahead options for one field (`v1/filter-options/<field>`). */
    search: (field: string, query: string) =>
      ["filter-options", "search", field, query] as const,
  },
  taxonomy: {
    /** Clause taxonomy tree; "tax" is the tax-clause taxonomy. */
    byKind: (kind: TaxonomyKind) =>
      kind === "tax" ? (["taxonomy", "tax"] as const) : (["taxonomy", "main"] as const),
    /** Corpus count per node (`v1/taxonomy/counts`, `.../tax-clauses/counts`). */
    counts: (kind: TaxonomyKind) => ["taxonomy", kind, "counts"] as const,
    /** One example section (or tax clause) tagged with a node. */
    example: (kind: TaxonomyKind, standardId: string) =>
      ["taxonomy", kind, "example", standardId] as const,
  },
  agreement: {
    detail: (uuid: string, focusSectionUuid?: string | null) =>
      focusSectionUuid
        ? (["agreement", uuid, "focus", focusSectionUuid] as const)
        : (["agreement", uuid] as const),
    sections: (uuid: string) => ["agreement", uuid, "sections"] as const,
    /** Smart-reader index; anonymous callers get it redacted around the focus. */
    readerIndex: (uuid: string, focusSectionUuid?: string | null) =>
      ["agreement", uuid, "reader-index", focusSectionUuid ?? null] as const,
  },
  agreements: {
    summary: ["agreements", "summary"] as const,
    recent: (limit: number) => ["agreements", "recent", limit] as const,
    index: (params: {
      page: number;
      pageSize: number;
      sortBy: string;
      sortDir: string;
      query: string;
    }) => ["agreements", "index", params] as const,
    statusSummary: ["agreements", "status-summary"] as const,
    dealTypesSummary: ["agreements", "deal-types-summary"] as const,
    trends: ["agreements", "trends"] as const,
  },
  counselLeaderboards: ["counsel-leaderboards"] as const,
  dumps: ["dumps"] as const,
  sections: {
    search: (params: Record<string, unknown>) =>
      ["sections", "search", params] as const,
  },
  taxClauses: {
    search: (params: Record<string, unknown>) =>
      ["tax-clauses", "search", params] as const,
  },
  transactions: {
    search: (params: Record<string, unknown>) =>
      ["transactions", "search", params] as const,
  },
  favorites: {
    all: ["favorites"] as const,
    tags: ["favorites", "tags"] as const,
    projects: ["favorites", "projects"] as const,
  },
} as const;
