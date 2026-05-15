import { useQuery } from "@tanstack/react-query";
import { FilterOptionsResponse } from "@shared/sections";
import { apiUrl } from "@/lib/api-config";
import { trackEvent } from "@/lib/analytics";
import { authFetch } from "@/lib/auth-fetch";
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
): Promise<FilterOptionsResponse> {
  const params = new URLSearchParams();
  fields?.forEach((field) => params.append("fields", field));
  const endpoint = params.size
    ? `v1/filter-options?${params.toString()}`
    : "v1/filter-options";

  const response = await authFetch(apiUrl(endpoint));
  if (!response.ok) {
    trackEvent("api_error", {
      endpoint: "api/filter-options",
      status: response.status,
      status_text: response.statusText,
    });
    throw new Error(`HTTP ${response.status}: ${response.statusText}`);
  }
  return (await response.json()) as FilterOptionsResponse;
}

export function useFilterOptions(
  options: UseFilterOptionsOptions = {},
): UseFilterOptionsReturn {
  const { enabled = true, fields } = options;

  const query = useQuery({
    queryKey: keys.filterOptions.list(fields),
    queryFn: () => fetchFilterOptions(fields),
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
    error: query.error ? (query.error as Error).message : null,
  };
}
