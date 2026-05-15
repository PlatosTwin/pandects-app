import { useQuery } from "@tanstack/react-query";
import type { TransactionSearchResponse } from "@shared/transactions";
import { apiUrl } from "@/lib/api-config";
import { authFetch } from "@/lib/auth-fetch";
import { IS_SERVER_RENDER } from "@/lib/query-client";

/**
 * Fetches the N most-recently-filed agreements. Used on the landing page to
 * show real, clickable previews of what's in the dataset.
 *
 * Reuses the agreement-search endpoint with no filters; sort_by=year is keyed
 * to filing_date server-side, so this returns the freshest entries.
 */
async function fetchRecentAgreements(
  limit: number,
): Promise<TransactionSearchResponse> {
  const params = new URLSearchParams();
  params.set("sort_by", "year");
  params.set("sort_direction", "desc");
  params.set("page", "1");
  params.set("page_size", String(limit));
  const res = await authFetch(
    apiUrl(`v1/search/agreements?${params.toString()}`),
  );
  if (!res.ok) {
    throw new Error(`Recent agreements request failed (${res.status})`);
  }
  return (await res.json()) as TransactionSearchResponse;
}

export function useRecentAgreements(limit: number = 4) {
  const query = useQuery({
    queryKey: ["recent-agreements", limit] as const,
    queryFn: () => fetchRecentAgreements(limit),
    enabled: !IS_SERVER_RENDER,
    staleTime: 5 * 60 * 1000,
    gcTime: 30 * 60 * 1000,
  });

  return {
    results: query.data?.results ?? [],
    isLoading: query.isLoading,
    error: query.error
      ? query.error instanceof Error
        ? query.error.message
        : "Failed to load recent agreements."
      : null,
  };
}
