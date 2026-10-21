/**
 * React Query hooks for the dataset pages: Agreement Index (overview charts),
 * Leaderboards, Trends & Analyses, and Bulk Data dumps.
 */
import { useEffect } from "react";
import { useQuery } from "@tanstack/react-query";
import type {
  AgreementDealTypeSummaryResponse,
  AgreementStatusSummaryResponse,
  AgreementTrendsResponse,
  CounselLeaderboardResponse,
  DumpInfo,
} from "@shared/insights";
import { trackEvent } from "@/lib/analytics";
import { apiGet, errorMessage, isApiError } from "@/lib/api/client";
import { logger } from "@/lib/logger";
import { IS_SERVER_RENDER } from "@/lib/query-client";
import { keys } from "@/lib/query-keys";

/** Aggregates rebuilt at most daily: reuse them across navigations. */
const AGGREGATE_STALE_TIME_MS = 5 * 60 * 1000;
const AGGREGATE_GC_TIME_MS = 30 * 60 * 1000;

function queryError(error: unknown, fallback: string): string | null {
  return error ? errorMessage(error, fallback) : null;
}

/** Processing-status chart data (`v1/agreements-status-summary`). */
export function useAgreementStatusSummary() {
  const query = useQuery({
    queryKey: keys.agreements.statusSummary,
    queryFn: ({ signal }) =>
      apiGet<AgreementStatusSummaryResponse>("v1/agreements-status-summary", undefined, {
        signal,
      }),
    enabled: !IS_SERVER_RENDER,
    staleTime: AGGREGATE_STALE_TIME_MS,
    gcTime: AGGREGATE_GC_TIME_MS,
  });
  return {
    data: query.data ?? null,
    isLoading: query.isPending && query.fetchStatus !== "idle",
    error: queryError(query.error, "Unable to load staging summary."),
  };
}

/** Deal-type chart data (`v1/agreements-deal-types-summary`); fetched once `enabled`. */
export function useAgreementDealTypesSummary({ enabled = true }: { enabled?: boolean } = {}) {
  const query = useQuery({
    queryKey: keys.agreements.dealTypesSummary,
    queryFn: ({ signal }) =>
      apiGet<AgreementDealTypeSummaryResponse>("v1/agreements-deal-types-summary", undefined, {
        signal,
      }),
    enabled: enabled && !IS_SERVER_RENDER,
    staleTime: AGGREGATE_STALE_TIME_MS,
    gcTime: AGGREGATE_GC_TIME_MS,
  });
  return {
    data: query.data ?? null,
    isLoading: query.isPending && query.fetchStatus !== "idle",
    isLoaded: query.isSuccess,
    error: queryError(query.error, "Unable to load deal type summary."),
  };
}

/** `v1/counsel-leaderboards`. Cached for back-navigation (replaces the old sessionStorage cache). */
export function useCounselLeaderboards() {
  const query = useQuery({
    queryKey: keys.counselLeaderboards,
    queryFn: ({ signal }) =>
      apiGet<CounselLeaderboardResponse>("v1/counsel-leaderboards", undefined, { signal }),
    enabled: !IS_SERVER_RENDER,
    staleTime: AGGREGATE_STALE_TIME_MS,
    gcTime: AGGREGATE_GC_TIME_MS,
  });
  return {
    data: query.data ?? null,
    isLoading: query.isPending,
    error: queryError(query.error, "Unable to load counsel leaderboards."),
    /** The raw ApiError, so ErrorState can map its status. */
    apiError: query.error ?? null,
    refetch: query.refetch,
  };
}

/** `v1/agreement-trends`. Cached for back-navigation (replaces the old sessionStorage cache). */
export function useAgreementTrends() {
  const query = useQuery({
    queryKey: keys.agreements.trends,
    queryFn: ({ signal }) =>
      apiGet<AgreementTrendsResponse>("v1/agreement-trends", undefined, { signal }),
    enabled: !IS_SERVER_RENDER,
    staleTime: AGGREGATE_STALE_TIME_MS,
    gcTime: AGGREGATE_GC_TIME_MS,
  });
  return {
    data: query.data ?? null,
    isLoading: query.isPending,
    error: queryError(query.error, "Unable to load trend analyses."),
    /** The raw ApiError, so ErrorState can map its status. */
    apiError: query.error ?? null,
    refetch: query.refetch,
  };
}

export interface DumpsData {
  /** "latest" first, then newest first. */
  dumps: DumpInfo[];
  latestSha256: string | null;
  latestSqlUrl: string | null;
}

function organizeDumps(data: DumpInfo[]): DumpsData {
  const latest = data.find((dump) => dump.timestamp === "latest");
  const dumps = [...data].sort((a, b) => {
    if (a.timestamp === "latest") return -1;
    if (b.timestamp === "latest") return 1;
    return new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime();
  });
  return {
    dumps,
    latestSha256: latest?.sha256 || null,
    latestSqlUrl: latest?.sql || null,
  };
}

const EMPTY_DUMPS: DumpsData = { dumps: [], latestSha256: null, latestSqlUrl: null };

/** Bulk-data dump listing (`v1/dumps`). */
export function useDumps() {
  const query = useQuery({
    queryKey: keys.dumps,
    queryFn: ({ signal }) => apiGet<DumpInfo[]>("v1/dumps", undefined, { signal }),
    enabled: !IS_SERVER_RENDER,
    select: organizeDumps,
  });

  useEffect(() => {
    const error = query.error;
    if (!error) return;
    if (import.meta.env.DEV) logger.error("Error fetching dumps:", error);
    if (isApiError(error) && !error.isNetworkError) {
      trackEvent("api_error", {
        endpoint: "v1/dumps",
        status: error.status,
        status_text: error.statusText,
      });
    }
    trackEvent("api_error", {
      endpoint: "v1/dumps",
      kind: isApiError(error) && error.isNetworkError ? "network" : "unknown",
    });
  }, [query.error]);

  return {
    ...(query.data ?? EMPTY_DUMPS),
    isLoading: query.isPending || (query.isFetching && query.isError),
    error: queryError(query.error, "Failed to load dumps"),
    /** The raw ApiError, so ErrorState can map its status. */
    apiError: query.error ?? null,
    refetch: query.refetch,
  };
}
