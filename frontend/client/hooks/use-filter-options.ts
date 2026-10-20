import { useCallback } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { FilterOptionsResponse } from "@shared/sections";
import { trackEvent } from "@/lib/analytics";
import { apiGet, errorMessage, isApiError } from "@/lib/api/client";
import { IS_SERVER_RENDER } from "@/lib/query-client";
import { keys } from "@/lib/query-keys";
import type { ClauseTypeTree } from "@/lib/clause-types";

interface UseFilterOptionsReturn {
  targets: string[];
  acquirers: string[];
  target_counsels: string[];
  acquirer_counsels: string[];
  target_industries: string[];
  acquirer_industries: string[];
  clause_types: ClauseTypeTree;
  isLoading: boolean;
  error: string | null;
}

interface UseFilterOptionsOptions {
  enabled?: boolean;
  fields?: ReadonlyArray<keyof FilterOptionsResponse>;
}

const EMPTY: UseFilterOptionsReturn = {
  targets: [],
  acquirers: [],
  target_counsels: [],
  acquirer_counsels: [],
  target_industries: [],
  acquirer_industries: [],
  clause_types: {},
  isLoading: false,
  error: null,
};

function parseClauseTypes(
  value: FilterOptionsResponse["clause_types"],
): ClauseTypeTree {
  if (value && typeof value === "object" && !Array.isArray(value)) {
    return value as ClauseTypeTree;
  }
  return {};
}

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

export function useFilterOptions(
  options: UseFilterOptionsOptions = {},
): UseFilterOptionsReturn {
  const { enabled = true, fields } = options;

  const query = useQuery({
    queryKey: keys.filterOptions.list(fields),
    queryFn: ({ signal }) => fetchFilterOptions(fields, signal),
    enabled: enabled && !IS_SERVER_RENDER,
    // Filter options change rarely (taxonomy/static-ish). Keep them fresh for
    // 30 minutes; the previous implementation cached forever in sessionStorage.
    staleTime: 30 * 60 * 1000,
    gcTime: 60 * 60 * 1000,
  });

  if (!enabled) return EMPTY;

  const data = query.data;
  return {
    targets: data?.targets ?? [],
    acquirers: data?.acquirers ?? [],
    target_counsels: data?.target_counsels ?? [],
    acquirer_counsels: data?.acquirer_counsels ?? [],
    target_industries: data?.target_industries ?? [],
    acquirer_industries: data?.acquirer_industries ?? [],
    clause_types: parseClauseTypes(data?.clause_types),
    isLoading: query.isLoading,
    error: query.error ? errorMessage(query.error, "Unable to load filter options.") : null,
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
