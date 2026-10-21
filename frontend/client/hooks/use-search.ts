import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { keepPreviousData, useQuery } from "@tanstack/react-query";
import type { SearchFilters } from "@shared/sections";
import type { SearchMode, SearchQueryFilters } from "@shared/search";
import { ApiError, apiGet, isApiError } from "@/lib/api/client";
import { trackEvent } from "@/lib/analytics";
import { DEFAULT_PAGE, DEFAULT_PAGE_SIZE } from "@/lib/constants";
import {
  downloadCsvFile,
  exportAllPagesForCsv,
  notifyCsvFailed,
  notifyCsvTruncated,
  type CsvExportRows,
} from "@/lib/csv-export";
import { logger } from "@/lib/logger";
import { IS_SERVER_RENDER } from "@/lib/query-client";
import { createEmptySearchFilters } from "@/lib/url-params";

export type SortField = "year" | "target" | "acquirer";
export type SortDirection = "asc" | "desc";

type AnalyticsParams = NonNullable<Parameters<typeof trackEvent>[1]>;

export type SearchAccessTier = "anonymous" | "user" | "api_key";

/** The paged response every search endpoint returns (server snake_case). */
export interface SearchPageResponse<TResult> {
  results: TResult[];
  access: { tier: SearchAccessTier; message?: string | null };
  total_count: number;
  total_count_is_approximate?: boolean;
  page: number;
  page_size: number;
  total_pages: number;
  has_next: boolean;
  has_prev: boolean;
}

/** The submitted search a fetch is keyed on. */
export interface CommittedSearch {
  filters: SearchQueryFilters;
  sortBy: SortField | null;
  sortDirection: SortDirection;
  /**
   * Bumped on every commit so the query key is distinct even when filters and
   * sort are unchanged: re-clicking Search after a failure must refetch
   * instead of returning the cached error.
   */
  nonce: number;
}

export type SearchSnapshot = Omit<CommittedSearch, "nonce">;

export interface SearchCsvConfig<TResult> {
  /** File name stem for "export all"; a date suffix is added. */
  filenameStem: string;
  /** File name stem when exporting selected rows only. */
  selectedFilenameStem: string;
  headers: string[];
  /** One CSV line (already escaped) per result. */
  toRow: (result: TResult) => string;
  /** Logger prefix when the export fetch fails. */
  failureLogLabel: string;
}

export interface SearchTelemetryConfig {
  /** `endpoint` label on `api_error` events; matches the request path. */
  endpointLabel: string;
  /** Send `api_error` with status for non-2xx search responses. */
  trackHttpErrors: boolean;
  /** Send `api_error` with kind for terminal search failures. */
  trackQueryFailures: boolean;
  /** Fired when a Search/Enter commit is made (markAsSearched). */
  performed?: {
    event: string;
    payload: (snapshot: SearchSnapshot) => AnalyticsParams;
  };
  /** Fired once when the results of a Search/Enter commit arrive. */
  resultsLoadedEvent?: string;
  /** Fired when a CSV file is built. */
  csvDownload?: {
    event: string;
    /** Extra fields from the exported query's filters. */
    payload?: (filters: SearchQueryFilters) => AnalyticsParams;
  };
}

/**
 * Everything that differs between search modes. Define configs at module
 * level: the hook memoizes on the config's identity.
 */
export interface SearchEndpointConfig<TResult> {
  mode: SearchMode;
  /** API path, e.g. "v1/sections". */
  endpoint: string;
  /** React Query key for a committed query string + nonce (from lib/query-keys). */
  queryKey: (params: { q: string; n: number }) => readonly unknown[];
  /** API query params for a submitted search (filters, sort, mode extras). */
  buildApiParams: (snapshot: SearchSnapshot) => URLSearchParams;
  getResultId: (result: TResult) => string;
  /**
   * When true, a 404 means "no results": it is not retried and does not open
   * the error modal.
   */
  silentNotFound: boolean;
  /** Logger prefix when a search fetch fails. */
  logLabel: string;
  telemetry: SearchTelemetryConfig;
  csv: SearchCsvConfig<TResult>;
}

/** Thrown for 404 responses when the endpoint treats "no results" as silent. */
export class SearchNotFoundError extends Error {
  constructor() {
    super("NOT_FOUND");
    this.name = "SearchNotFoundError";
  }
}

export const SEARCH_NETWORK_ERROR_MESSAGE =
  "Network error: unable to reach the back end database. Check your connection and try again.";

/** User-facing text for a failed search, by status. */
export function describeSearchError(error: unknown): string {
  if (!isApiError(error) || error.isNetworkError) return SEARCH_NETWORK_ERROR_MESSAGE;
  const server = error.serverMessage;
  const { status } = error;
  if (status === 400 || status === 422) {
    return server
      ? `The search could not run: ${server}`
      : "The search could not run because a filter value is invalid. Adjust the filters and try again.";
  }
  if (status === 401) {
    return "Your session has expired. Sign in again, then rerun the search.";
  }
  if (status === 403) {
    return server ?? "You don't have access to these search results.";
  }
  if (status === 404) {
    return "The search service could not be found. Try again in a moment.";
  }
  if (status === 429) {
    return server
      ? `Too many searches in a short time: ${server} Wait a moment and try again.`
      : "Too many searches in a short time. Wait a moment and try again.";
  }
  if (status >= 500) {
    return `The search service had a problem (HTTP ${status}). Try again in a moment.`;
  }
  return server ?? `The search failed (HTTP ${status}). Try again.`;
}

function failureKind(error: unknown): "network" | "unknown" {
  return isApiError(error) && error.isNetworkError ? "network" : "unknown";
}

export interface PerformSearchOptions {
  /** Reset to page 1 (also in the draft, unless `filters` is given). */
  resetPage?: boolean;
  /**
   * A user-initiated Search/Enter: fires the `performed` and
   * `resultsLoaded` telemetry. Defaults to `resetPage`.
   */
  markAsSearched?: boolean;
  /** Search these filters instead of the hook's own draft. */
  filters?: SearchQueryFilters;
  /** Sort to submit; defaults to the hook's sort state. `null` = unsorted. */
  sortBy?: SortField | null;
  sortDirection?: SortDirection;
}

type ArrayFilterField = {
  [K in keyof SearchFilters]-?: NonNullable<SearchFilters[K]> extends string[] ? K : never;
}[keyof SearchFilters];
type TextFilterField = "agreement_uuid" | "section_uuid";

export type CsvScope = "auto" | "all" | "selected";

const EMPTY_ACCESS: SearchPageResponse<unknown>["access"] = { tier: "anonymous" };

/**
 * Only a failure that never reached the API (network error) or a fast proxy
 * failure (502/503: a machine starting or stopping) is retried. A 500/504 is
 * usually a query that already ran into the server's statement limit;
 * retrying it holds a server worker and a database connection for another
 * minute, and the server keeps running a search after the browser drops it.
 */
function isRetryableSearchError(error: Error): boolean {
  if (!isApiError(error)) return false;
  return error.isNetworkError || error.status === 502 || error.status === 503;
}

/**
 * One committed-search hook for every mode (sections, tax clauses, deals).
 *
 * - `filters` / `sortBy` / `sortDirection` are the draft the sidebar edits;
 *   editing them never fetches.
 * - `actions.performSearch` (and `goToPage`, `changePageSize`) commit a
 *   snapshot; a `useQuery` keyed on it fetches the page. Pass `filters` /
 *   `sortBy` / `sortDirection` to search something other than this hook's
 *   own draft (a page that shares one draft across modes does this).
 * - The previous page stays rendered while the next one loads.
 * - Selection is cleared on every commit; CSV export uses the last commit
 *   whose results actually loaded.
 */
export function useSearch<TResult>(config: SearchEndpointConfig<TResult>) {
  const [filters, setFilters] = useState<SearchQueryFilters>(createEmptySearchFilters);
  const [sortBy, setSortBy] = useState<SortField | null>("year");
  const [sortDirection, setSortDirection] = useState<SortDirection>("desc");
  const [committed, setCommitted] = useState<CommittedSearch | null>(null);
  const [selectedResults, setSelectedResults] = useState<Set<string>>(() => new Set());
  const [showErrorModal, setShowErrorModal] = useState(false);
  const [errorMessage, setErrorMessage] = useState("");
  const nonceRef = useRef(0);
  const fireResultsLoadedRef = useRef(false);

  const committedQueryString = useMemo(
    () => (committed ? config.buildApiParams(committed).toString() : ""),
    [committed, config],
  );

  const query = useQuery<SearchPageResponse<TResult>>({
    queryKey: config.queryKey({ q: committedQueryString, n: committed?.nonce ?? 0 }),
    enabled: !IS_SERVER_RENDER && committed !== null,
    staleTime: 60 * 1000,
    // Each commit gets a fresh key (see CommittedSearch.nonce); keep the
    // previous commit's results on screen while the next fetch is in flight.
    placeholderData: keepPreviousData,
    retry: (failureCount: number, error: Error) =>
      failureCount < 1 && isRetryableSearchError(error),
    queryFn: async ({ signal }) => {
      try {
        return await apiGet<SearchPageResponse<TResult>>(
          config.endpoint,
          new URLSearchParams(committedQueryString),
          { signal },
        );
      } catch (error) {
        if (error instanceof ApiError && !error.isNetworkError) {
          if (config.silentNotFound && error.status === 404) {
            throw new SearchNotFoundError();
          }
          if (config.telemetry.trackHttpErrors) {
            trackEvent("api_error", {
              endpoint: config.telemetry.endpointLabel,
              status: error.status,
              status_text: error.statusText,
            });
          }
        }
        throw error;
      }
    },
  });

  // placeholderData also fills the disabled query after `clearResults`, so
  // gate on an active commit to keep "no search yet" truly empty.
  const response = committed !== null ? query.data ?? null : null;
  const results = useMemo(() => response?.results ?? [], [response]);
  const access = response?.access ?? EMPTY_ACCESS;
  const error =
    query.error && !(query.error instanceof SearchNotFoundError) ? query.error : null;

  // The last commit whose results actually loaded. While a newer fetch is in
  // flight or has failed, the screen still shows this one's results, so
  // exports use it rather than `committed`.
  const loadedCommittedRef = useRef<CommittedSearch | null>(null);
  useEffect(() => {
    if (committed === null) {
      loadedCommittedRef.current = null;
    } else if (query.isSuccess && !query.isPlaceholderData) {
      loadedCommittedRef.current = committed;
    }
  }, [committed, query.isSuccess, query.isPlaceholderData]);
  const getLoadedCommitted = useCallback(() => loadedCommittedRef.current, []);

  // Surface terminal errors via the error modal.
  useEffect(() => {
    if (!error) return;
    logger.error(config.logLabel, error);
    if (config.telemetry.trackQueryFailures) {
      trackEvent("api_error", {
        endpoint: config.telemetry.endpointLabel,
        kind: failureKind(error),
      });
    }
    setErrorMessage(describeSearchError(error));
    setShowErrorModal(true);
  }, [error, config]);

  // Results-loaded telemetry, once per Search/Enter commit.
  useEffect(() => {
    const event = config.telemetry.resultsLoadedEvent;
    if (!response || !fireResultsLoadedRef.current || !committed || query.isPlaceholderData) {
      return;
    }
    fireResultsLoadedRef.current = false;
    if (!event) return;
    trackEvent(event, {
      total_count: response.total_count,
      total_pages: response.total_pages,
      page: committed.filters.page,
      page_size: committed.filters.page_size,
    });
  }, [response, committed, config, query.isPlaceholderData]);

  // Mirror the fetched page into the draft so pagination controls and URL
  // sync follow what was actually fetched. The page size is not mirrored: the
  // API clamps it to the caller's tier, and writing that clamp into the draft
  // would put it in shareable URLs. The requested size stays in the draft
  // (and URL); `pageSize` below reports the size the API actually used.
  useEffect(() => {
    if (!response || query.isPlaceholderData) return;
    setFilters((prev) => (prev.page === response.page ? prev : { ...prev, page: response.page }));
  }, [response, query.isPlaceholderData]);

  // Every commit hides the error modal and clears the row selection.
  const commit = useCallback((next: SearchSnapshot) => {
    setShowErrorModal(false);
    setSelectedResults(new Set());
    nonceRef.current += 1;
    setCommitted({ ...next, nonce: nonceRef.current });
  }, []);

  /** Drop the committed search (back to "not searched yet"); keeps the draft. */
  const clearResults = useCallback(() => {
    setCommitted(null);
    setSelectedResults(new Set());
  }, []);

  const updateFilter = useCallback(
    <K extends keyof SearchQueryFilters>(field: K, value: SearchQueryFilters[K]) => {
      setFilters((prev) => ({ ...prev, [field]: value }));
    },
    [],
  );

  const hydrateFilters = useCallback((next: Partial<SearchQueryFilters>) => {
    setFilters((prev) => ({
      ...prev,
      ...next,
      page: next.page ?? prev.page,
      page_size: next.page_size ?? prev.page_size,
    }));
  }, []);

  const toggleFilterValue = useCallback((field: keyof SearchFilters, value: string) => {
    if (field === "page" || field === "page_size") return;
    if (field === "agreement_uuid" || field === "section_uuid") return;
    const arrayField = field as ArrayFilterField;
    setFilters((prev) => {
      const current = prev[arrayField] ?? [];
      const nextValues = current.includes(value)
        ? current.filter((v) => v !== value)
        : [...current, value];
      return { ...prev, [arrayField]: nextValues };
    });
  }, []);

  const setTextFilterValue = useCallback((field: keyof SearchFilters, value: string) => {
    if (field === "page" || field === "page_size") return;
    const trimmed = value.trim();
    setFilters((prev) => ({ ...prev, [field as TextFilterField]: trimmed || undefined }));
  }, []);

  const setIncludeRepWarranty = useCallback((value: boolean) => {
    setFilters((prev) => {
      if (value) return { ...prev, include_rep_warranty: true };
      const { include_rep_warranty: _omit, ...rest } = prev;
      void _omit;
      return rest;
    });
  }, []);

  const toggleSortDirection = useCallback(() => {
    setSortDirection((d) => (d === "asc" ? "desc" : "asc"));
  }, []);

  const performSearch = useCallback(
    async (options: PerformSearchOptions = {}) => {
      const { resetPage = false, markAsSearched = resetPage } = options;
      const base = options.filters ?? filters;
      const effective: SearchQueryFilters = resetPage ? { ...base, page: 1 } : base;
      if (resetPage && !options.filters) {
        setFilters((prev) => ({ ...prev, page: 1 }));
      }
      const snapshot: SearchSnapshot = {
        filters: effective,
        sortBy: options.sortBy !== undefined ? options.sortBy : sortBy,
        sortDirection: options.sortDirection ?? sortDirection,
      };

      if (markAsSearched) {
        fireResultsLoadedRef.current = true;
        const performed = config.telemetry.performed;
        if (performed) trackEvent(performed.event, performed.payload(snapshot));
      }

      commit(snapshot);
    },
    [filters, sortBy, sortDirection, config, commit],
  );

  const goToPage = useCallback(
    async (page: number) => {
      const next: SearchQueryFilters = { ...filters, page };
      setFilters(next);
      await performSearch({ filters: next });
    },
    [filters, performSearch],
  );

  const changePageSize = useCallback(
    async (pageSize: number) => {
      const next: SearchQueryFilters = { ...filters, page_size: pageSize, page: 1 };
      setFilters(next);
      await performSearch({ filters: next });
    },
    [filters, performSearch],
  );

  /** Reset the draft to empty and drop the committed search. */
  const clearFilters = useCallback(() => {
    setFilters({
      ...createEmptySearchFilters(),
      agreement_uuid: undefined,
      section_uuid: undefined,
    });
    clearResults();
  }, [clearResults]);

  const closeErrorModal = useCallback(() => {
    setShowErrorModal(false);
    setErrorMessage("");
  }, []);

  const toggleResultSelection = useCallback((resultId: string) => {
    setSelectedResults((prev) => {
      const next = new Set(prev);
      if (next.has(resultId)) next.delete(resultId);
      else next.add(resultId);
      return next;
    });
  }, []);

  const toggleSelectAll = useCallback(() => {
    const getId = config.getResultId;
    setSelectedResults((prev) => {
      const allSelected = results.every((r) => prev.has(getId(r)));
      const next = new Set(prev);
      if (allSelected) results.forEach((r) => next.delete(getId(r)));
      else results.forEach((r) => next.add(getId(r)));
      return next;
    });
  }, [results, config]);

  const clearSelection = useCallback(() => {
    setSelectedResults(new Set());
  }, []);

  /**
   * Build a CSV. `"auto"` (the default) exports the selection when rows are
   * selected, else every result; `"all"` / `"selected"` force one or the other.
   */
  const downloadCSV = useCallback(async (scope: CsvScope = "auto") => {
    if (scope === "selected" && selectedResults.size === 0) return;
    const isSelection = scope === "selected" || (scope === "auto" && selectedResults.size > 0);
    let rows: TResult[] = [];
    let exportedFrom: CsvExportRows<TResult> | null = null;

    if (isSelection) {
      rows = results.filter((r) => selectedResults.has(config.getResultId(r)));
    } else {
      // Export what the results show: the last submitted query that loaded,
      // not the sidebar draft.
      const loaded = getLoadedCommitted();
      if (!loaded) return;
      try {
        const exported = await exportAllPagesForCsv<TResult>(
          config.endpoint,
          config.buildApiParams(loaded).toString(),
        );
        if (!exported) return;
        exportedFrom = exported;
        rows = exported.rows;
      } catch (exportError) {
        logger.error(config.csv.failureLogLabel, exportError);
        trackEvent("api_error", {
          endpoint: config.telemetry.endpointLabel,
          kind: failureKind(exportError),
        });
        notifyCsvFailed();
        return;
      }
    }

    if (rows.length === 0) return;

    const csvDownload = config.telemetry.csvDownload;
    if (csvDownload) {
      const exportedFilters = getLoadedCommitted()?.filters ?? filters;
      trackEvent(csvDownload.event, {
        mode: isSelection ? "selected" : "all",
        selected_count: isSelection ? selectedResults.size : 0,
        downloaded_count: rows.length,
        truncated: exportedFrom?.truncated ?? false,
        ...(csvDownload.payload?.(exportedFilters) ?? {}),
      });
    }

    downloadCsvFile(
      isSelection ? config.csv.selectedFilenameStem : config.csv.filenameStem,
      config.csv.headers,
      rows.map(config.csv.toRow),
    );

    if (exportedFrom?.truncated) {
      notifyCsvTruncated(
        rows.length,
        exportedFrom.totalCount,
        exportedFrom.totalCountIsApproximate,
        access.tier === "anonymous",
      );
    }
  }, [selectedResults, results, config, getLoadedCommitted, filters, access.tier]);

  return {
    mode: config.mode,
    // Draft
    filters,
    sortBy,
    sortDirection,
    currentPage: filters.page || DEFAULT_PAGE,
    /** Rows per page as fetched (tier-clamped), else the requested size. */
    pageSize: response?.page_size || filters.page_size || DEFAULT_PAGE_SIZE,
    // Committed search + results
    committed,
    hasSearched: committed !== null,
    isSearching: query.isFetching,
    error,
    results,
    totalCount: response?.total_count ?? 0,
    totalCountIsApproximate: response?.total_count_is_approximate ?? false,
    totalPages: response?.total_pages ?? 0,
    hasNext: response?.has_next ?? false,
    hasPrev: response?.has_prev ?? false,
    access,
    // Selection + error modal
    selectedResults,
    showErrorModal,
    errorMessage,
    actions: {
      updateFilter,
      hydrateFilters,
      toggleFilterValue,
      setTextFilterValue,
      setIncludeRepWarranty,
      setSortBy,
      setSortDirection,
      toggleSortDirection,
      performSearch,
      goToPage,
      changePageSize,
      clearFilters,
      clearResults,
      downloadCSV,
      closeErrorModal,
      toggleResultSelection,
      toggleSelectAll,
      clearSelection,
    },
  };
}

export type UseSearchResult<TResult> = ReturnType<typeof useSearch<TResult>>;
