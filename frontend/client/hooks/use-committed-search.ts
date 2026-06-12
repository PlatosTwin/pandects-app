import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { authFetch } from "@/lib/auth-fetch";
import { logger } from "@/lib/logger";
import { IS_SERVER_RENDER } from "@/lib/query-client";
import type { ClauseTypeTree } from "@/lib/clause-types";

export type SortField = "year" | "target" | "acquirer";
export type SortDirection = "asc" | "desc";

export interface CommittedQuery<TFilters> {
  filters: TFilters;
  clauseTypesNested?: ClauseTypeTree;
  sortBy: SortField | null;
  sortDirection: SortDirection;
  // Bumped on every commit so the queryKey is distinct even when filters/sort
  // are unchanged from the previous attempt. Without this, a Search re-click
  // after a failed fetch wouldn't refetch — React Query would just return the
  // cached error for the same key.
  nonce: number;
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

interface CommittedSearchCoreConfig<TFilters, TResponse, TResult> {
  /** Serialize a committed snapshot into the request query string. */
  buildQueryString: (committed: CommittedQuery<TFilters>) => string;
  /** Full request URL for a given query string. */
  fetchUrl: (queryString: string) => string;
  /** React Query key for a committed query string + nonce. */
  queryKey: (params: { q: string; n: number }) => readonly unknown[];
  getResults: (response: TResponse) => TResult[];
  getResultId: (result: TResult) => string;
  /**
   * When true, a 404 throws SearchNotFoundError: not retried, no error modal,
   * prior results preserved. When false, 404s behave like any other failure.
   */
  silentNotFound: boolean;
  /** Prefix for the logger line when a fetch fails. */
  logLabel: string;
  /** Telemetry for non-2xx responses (status known). Optional by design —
   * telemetry is a per-endpoint product decision, not core machinery. */
  trackHttpError?: (status: number, statusText: string) => void;
  /** Telemetry for terminal query failures (network/unknown). */
  trackFailure?: (error: unknown) => void;
}

/**
 * Shared state machine for committed searches (sections, tax clauses, deals).
 *
 * Owns the committed snapshot + nonce, the React Query fetch keyed on it, the
 * result-row selection set (cleared on each new commit), and the error modal.
 * Draft-filter editing, CSV export, sorting, and telemetry payloads stay in
 * the per-endpoint hooks.
 *
 * `config` must be a module-level constant: its functions are captured by
 * memoized callbacks and query options, so a per-render object would defeat
 * memoization and churn the query.
 */
export function useCommittedSearchCore<TFilters, TResponse, TResult>(
  config: CommittedSearchCoreConfig<TFilters, TResponse, TResult>,
) {
  const [committed, setCommitted] = useState<CommittedQuery<TFilters> | null>(
    null,
  );
  const [selectedResults, setSelectedResults] = useState<Set<string>>(
    new Set(),
  );
  const [showErrorModal, setShowErrorModal] = useState(false);
  const [errorMessage, setErrorMessage] = useState("");
  const nonceRef = useRef(0);

  const committedQueryString = useMemo(
    () => (committed ? config.buildQueryString(committed) : ""),
    [committed, config],
  );

  const query = useQuery<TResponse>({
    queryKey: config.queryKey({
      q: committedQueryString,
      n: committed?.nonce ?? 0,
    }),
    enabled: !IS_SERVER_RENDER && committed !== null,
    staleTime: 60 * 1000,
    ...(config.silentNotFound
      ? {
          retry: (failureCount: number, error: Error) =>
            !(error instanceof SearchNotFoundError) && failureCount < 1,
        }
      : {}),
    queryFn: async () => {
      const res = await authFetch(config.fetchUrl(committedQueryString));
      if (!res.ok) {
        if (config.silentNotFound && res.status === 404) {
          throw new SearchNotFoundError();
        }
        config.trackHttpError?.(res.status, res.statusText);
        throw new Error(`HTTP ${res.status}: ${res.statusText}`);
      }
      return (await res.json()) as TResponse;
    },
  });

  const responseData: TResponse | null = query.data ?? null;
  const results = useMemo(
    () => (responseData ? config.getResults(responseData) : []),
    [responseData, config],
  );

  // Clear selection whenever a new committed query is issued.
  useEffect(() => {
    if (committed !== null) {
      setSelectedResults(new Set());
    }
  }, [committed]);

  // Surface terminal errors via the error modal. SearchNotFoundError stays
  // silent and preserves prior results.
  useEffect(() => {
    if (!query.error) return;
    if (query.error instanceof SearchNotFoundError) return;
    logger.error(config.logLabel, query.error);
    config.trackFailure?.(query.error);
    setErrorMessage(SEARCH_NETWORK_ERROR_MESSAGE);
    setShowErrorModal(true);
  }, [query.error, config]);

  /** Commit a new snapshot: hides the error modal and bumps the nonce. */
  const commit = useCallback(
    (next: Omit<CommittedQuery<TFilters>, "nonce">) => {
      setShowErrorModal(false);
      nonceRef.current += 1;
      setCommitted({ ...next, nonce: nonceRef.current });
    },
    [],
  );

  /** Drop the committed snapshot (back to the not-yet-searched state). */
  const uncommit = useCallback(() => {
    setCommitted(null);
    setSelectedResults(new Set());
  }, []);

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
      if (allSelected) {
        results.forEach((r) => next.delete(getId(r)));
      } else {
        results.forEach((r) => next.add(getId(r)));
      }
      return next;
    });
  }, [results, config]);

  const clearSelection = useCallback(() => {
    setSelectedResults(new Set());
  }, []);

  return {
    committed,
    commit,
    uncommit,
    query,
    responseData,
    results,
    isSearching: query.isFetching,
    selectedResults,
    toggleResultSelection,
    toggleSelectAll,
    clearSelection,
    showErrorModal,
    errorMessage,
    closeErrorModal,
  };
}
