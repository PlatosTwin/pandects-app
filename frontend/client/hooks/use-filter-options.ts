import { useCallback } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { FilterOptionsResponse } from "@shared/sections";
import { trackEvent } from "@/lib/analytics";
import { apiGet, errorMessage, isApiError } from "@/lib/api/client";
import { IS_SERVER_RENDER } from "@/lib/query-client";
import { keys } from "@/lib/query-keys";
import type { ClauseTypeTree } from "@/lib/clause-types";
import { useTaxonomy } from "@/hooks/use-taxonomy";

interface UseFilterOptionsReturn {
  targets: string[];
  acquirers: string[];
  target_counsels: string[];
  acquirer_counsels: string[];
  target_industries: string[];
  acquirer_industries: string[];
  /** The `GET /v1/taxonomy` tree (shared `useTaxonomy` cache). */
  clause_types: ClauseTypeTree;
  /** NAICS label per industry code in `target_industries`/`acquirer_industries`. */
  industry_labels: Record<string, string>;
  isLoading: boolean;
  error: string | null;
}

interface UseFilterOptionsOptions {
  enabled?: boolean;
  fields?: ReadonlyArray<keyof FilterOptionsResponse>;
}

/** Every `/v1/filter-options` field except the deprecated `clause_types`. */
const SERVER_FILTER_FIELDS: ReadonlyArray<keyof FilterOptionsResponse> = [
  "targets",
  "acquirers",
  "target_counsels",
  "acquirer_counsels",
  "target_industries",
  "acquirer_industries",
];

const EMPTY: UseFilterOptionsReturn = {
  targets: [],
  acquirers: [],
  target_counsels: [],
  acquirer_counsels: [],
  target_industries: [],
  acquirer_industries: [],
  clause_types: {},
  industry_labels: {},
  isLoading: false,
  error: null,
};

async function fetchFilterOptions(
  fields?: ReadonlyArray<keyof FilterOptionsResponse>,
  signal?: AbortSignal,
): Promise<FilterOptionsResponse> {
  try {
    return await apiGet<FilterOptionsResponse>(
      "v1/filter-options",
      { fields: fields ? [...fields] : undefined },
      { signal },
    );
  } catch (error) {
    if (isApiError(error) && !error.isNetworkError) {
      trackEvent("api_error", {
        endpoint: "v1/filter-options",
        status: error.status,
        status_text: error.statusText,
      });
    }
    throw error;
  }
}

const EMPTY_CLAUSE_TYPES: ClauseTypeTree = {};
const EMPTY_LABELS: Record<string, string> = {};

/**
 * Agreement filter options. `clause_types` is served from the `/v1/taxonomy`
 * cache (`useTaxonomy`), never from `/v1/filter-options`, so the Taxonomy page,
 * Search, the reader and Favorites share one copy of the tree. When
 * `clause_types` is the only requested field, `/v1/filter-options` is not
 * called at all.
 */
export function useFilterOptions(
  options: UseFilterOptionsOptions = {},
): UseFilterOptionsReturn {
  const { enabled = true, fields } = options;
  const wantsClauseTypes = !fields || fields.includes("clause_types");
  const serverFields = fields?.filter((field) => field !== "clause_types");
  const needsServer = !serverFields || serverFields.length > 0;
  // With no `fields`, ask for every summary field except the taxonomy.
  const requestedFields = serverFields ?? SERVER_FILTER_FIELDS;

  const query = useQuery({
    queryKey: keys.filterOptions.list(requestedFields),
    queryFn: ({ signal }) => fetchFilterOptions(requestedFields, signal),
    enabled: enabled && needsServer && !IS_SERVER_RENDER,
    // Filter options change rarely (taxonomy/static-ish). Keep them fresh for
    // 30 minutes; the previous implementation cached forever in sessionStorage.
    staleTime: 30 * 60 * 1000,
    gcTime: 60 * 60 * 1000,
  });
  const taxonomy = useTaxonomy({ enabled: enabled && wantsClauseTypes });

  if (!enabled) return EMPTY;

  const data = query.data;
  const errorValue = query.error
    ? errorMessage(query.error, "Unable to load filter options.")
    : taxonomy.error;
  return {
    targets: data?.targets ?? [],
    acquirers: data?.acquirers ?? [],
    target_counsels: data?.target_counsels ?? [],
    acquirer_counsels: data?.acquirer_counsels ?? [],
    target_industries: data?.target_industries ?? [],
    acquirer_industries: data?.acquirer_industries ?? [],
    clause_types: (wantsClauseTypes && taxonomy.taxonomyTree) || EMPTY_CLAUSE_TYPES,
    industry_labels: data?.industry_labels ?? EMPTY_LABELS,
    isLoading: (needsServer && query.isLoading) || (wantsClauseTypes && taxonomy.isLoading),
    error: errorValue,
  };
}

export type FilterOptionSearchField = "target" | "acquirer";

const FILTER_OPTION_SEARCH_LIMIT = 100;

async function fetchFilterOptionSearch(
  field: FilterOptionSearchField,
  query: string,
): Promise<string[]> {
  const payload = await apiGet<{ options?: unknown }>(`v1/filter-options/${field}`, {
    query: query || undefined,
    limit: FILTER_OPTION_SEARCH_LIMIT,
  });
  return Array.isArray(payload.options)
    ? payload.options.filter((value): value is string => typeof value === "string")
    : [];
}

/**
 * Type-ahead loader for the target/acquirer pickers. Returns a stable
 * `(field, query) => Promise<string[]>`; results are cached per (field,
 * query) for five minutes, so reopening a picker or retyping a prefix does
 * not refetch.
 */
export function useFilterOptionSearch() {
  const queryClient = useQueryClient();
  return useCallback(
    (field: FilterOptionSearchField, rawQuery: string) => {
      const query = rawQuery.trim();
      return queryClient.fetchQuery({
        queryKey: keys.filterOptions.search(field, query),
        queryFn: () => fetchFilterOptionSearch(field, query),
        staleTime: 5 * 60 * 1000,
        gcTime: 30 * 60 * 1000,
      });
    },
    [queryClient],
  );
}
