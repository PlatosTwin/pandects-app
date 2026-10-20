import { useQuery } from "@tanstack/react-query";
import type { TransactionSearchResponse } from "@shared/transactions";
import { apiGet, errorMessage } from "@/lib/api/client";
import { IS_SERVER_RENDER } from "@/lib/query-client";
import { keys } from "@/lib/query-keys";

/**
 * Fetches the N most-recently-filed agreements. Used on the landing page to
 * show real, clickable previews of what's in the dataset.
 *
 * Reuses the agreement-search endpoint with no filters; sort_by=year is keyed
 * to filing_date server-side, so this returns the freshest entries.
 */
function fetchRecentAgreements(
  limit: number,
  signal?: AbortSignal,
): Promise<TransactionSearchResponse> {
  return apiGet<TransactionSearchResponse>(
    "v1/search/agreements",
    { sort_by: "year", sort_direction: "desc", page: 1, page_size: limit },
    { signal },
  );
}

export function useRecentAgreements(limit: number = 4) {
  const query = useQuery({
    queryKey: keys.agreements.recent(limit),
    queryFn: ({ signal }) => fetchRecentAgreements(limit, signal),
    enabled: !IS_SERVER_RENDER,
    staleTime: 5 * 60 * 1000,
    gcTime: 30 * 60 * 1000,
  });

  return {
    results: query.data?.results ?? [],
    isLoading: query.isLoading,
    error: query.error
      ? errorMessage(query.error, "Failed to load recent agreements.")
      : null,
  };
}
