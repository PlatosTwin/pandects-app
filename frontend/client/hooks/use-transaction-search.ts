import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { apiUrl } from "@/lib/api-config";
import { authFetch } from "@/lib/auth-fetch";
import { buildSearchParams } from "@/lib/url-params";
import { logger } from "@/lib/logger";
import { LARGE_PAGE_SIZE_FOR_CSV, DEFAULT_PAGE } from "@/lib/constants";
import { IS_SERVER_RENDER } from "@/lib/query-client";
import { keys } from "@/lib/query-keys";
import {
  formatCompactCurrencyValue,
  formatDateValue,
  formatEnumValue,
} from "@/lib/format-utils";
import type { ClauseTypeTree } from "@/lib/clause-types";
import type { SearchFilters } from "@shared/sections";
import type {
  TransactionSearchResponse,
  TransactionSearchResult,
} from "@shared/transactions";

interface CommittedQuery {
  filters: SearchFilters;
  clauseTypesNested?: ClauseTypeTree;
  sortBy: "year" | "target" | "acquirer" | null;
  sortDirection: "asc" | "desc";
  // Bumped on every commit so the queryKey is distinct even when filters/sort
  // are unchanged from the previous attempt. Without this, a Search re-click
  // after a failed fetch wouldn't refetch — React Query would just return the
  // cached error for the same key.
  nonce: number;
}

const EMPTY_ACCESS: TransactionSearchResponse["access"] = { tier: "anonymous" };

function buildTransactionQueryString(committed: CommittedQuery): string {
  const params = buildSearchParams(committed.filters, committed.clauseTypesNested);
  if (committed.sortBy) {
    params.set("sort_by", committed.sortBy);
    params.set("sort_direction", committed.sortDirection);
  }
  return params.toString();
}

/**
 * Declarative transaction (deal) search. Unlike `useTaxClauses` / `useSections`,
 * this hook does not own a filter draft — the parent passes filters in via
 * `performSearch(...)`. The hook only tracks the committed snapshot that drives
 * `useQuery`.
 */
export function useTransactionSearch() {
  const queryClient = useQueryClient();
  const [committed, setCommitted] = useState<CommittedQuery | null>(null);
  const [hasSearched, setHasSearched] = useState(false);
  const [selectedResults, setSelectedResults] = useState<Set<string>>(new Set());
  const [showErrorModal, setShowErrorModal] = useState(false);
  const [errorMessage, setErrorMessage] = useState("");
  const nonceRef = useRef(0);

  const committedQueryString = useMemo(
    () => (committed ? buildTransactionQueryString(committed) : ""),
    [committed],
  );

  const query = useQuery<TransactionSearchResponse>({
    queryKey: keys.transactions.search({
      q: committedQueryString,
      n: committed?.nonce ?? 0,
    }),
    enabled: !IS_SERVER_RENDER && committed !== null,
    staleTime: 60 * 1000,
    queryFn: async () => {
      const response = await authFetch(
        apiUrl(`v1/search/agreements?${committedQueryString}`),
      );
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }
      return (await response.json()) as TransactionSearchResponse;
    },
  });

  const responseData: TransactionSearchResponse | null = query.data ?? null;
  const results = responseData?.results ?? [];
  const totalCount = responseData?.total_count ?? 0;
  const totalCountIsApproximate = responseData?.total_count_is_approximate ?? false;
  const totalPages = responseData?.total_pages ?? 0;
  const hasNext = responseData?.has_next ?? false;
  const hasPrev = responseData?.has_prev ?? false;
  const access = responseData?.access ?? EMPTY_ACCESS;
  const isSearching = query.isFetching;

  // Clear selection whenever a new committed query is issued.
  useEffect(() => {
    if (committed !== null) {
      setSelectedResults(new Set());
    }
  }, [committed]);

  // Surface real errors via the error modal.
  useEffect(() => {
    if (!query.error) return;
    logger.error("Agreement search failed:", query.error);
    setErrorMessage(
      "Network error: unable to reach the back end database. Check your connection and try again.",
    );
    setShowErrorModal(true);
  }, [query.error]);

  const performSearch = useCallback(
    async ({
      filters,
      clauseTypesNested,
      sortBy,
      sortDirection,
      markAsSearched = true,
    }: {
      filters: SearchFilters;
      clauseTypesNested?: ClauseTypeTree;
      sortBy: "year" | "target" | "acquirer" | null;
      sortDirection: "asc" | "desc";
      markAsSearched?: boolean;
    }) => {
      setShowErrorModal(false);
      if (markAsSearched) setHasSearched(true);
      nonceRef.current += 1;
      setCommitted({
        filters,
        clauseTypesNested,
        sortBy,
        sortDirection,
        nonce: nonceRef.current,
      });
    },
    [],
  );

  const clear = useCallback(() => {
    setCommitted(null);
    setHasSearched(false);
    setSelectedResults(new Set());
  }, []);

  const toggleResultSelection = useCallback((agreementUuid: string) => {
    setSelectedResults((prev) => {
      const next = new Set(prev);
      if (next.has(agreementUuid)) next.delete(agreementUuid);
      else next.add(agreementUuid);
      return next;
    });
  }, []);

  const toggleSelectAll = useCallback(() => {
    setSelectedResults((prev) => {
      const allSelected = results.every((r) => prev.has(r.agreement_uuid));
      const next = new Set(prev);
      if (allSelected) {
        results.forEach((r) => next.delete(r.agreement_uuid));
      } else {
        results.forEach((r) => next.add(r.agreement_uuid));
      }
      return next;
    });
  }, [results]);

  const clearSelection = useCallback(() => {
    setSelectedResults(new Set());
  }, []);

  const csvEscape = (value: string | number | null | undefined): string => {
    if (value === null || value === undefined || value === "") return "";
    const str = String(value);
    return `"${str.replace(/"/g, '""')}"`;
  };

  const downloadCSV = useCallback(
    async (clauseTypesNested?: ClauseTypeTree, filters?: SearchFilters) => {
      let rows: TransactionSearchResult[] = [];

      if (selectedResults.size > 0) {
        rows = results.filter((r) => selectedResults.has(r.agreement_uuid));
      } else if (filters) {
        try {
          const searchFilters = {
            ...filters,
            page: undefined,
            page_size: undefined,
          };
          const params = buildSearchParams(searchFilters, clauseTypesNested, false);
          params.append("page_size", LARGE_PAGE_SIZE_FOR_CSV.toString());
          params.append("page", DEFAULT_PAGE.toString());
          const queryString = params.toString();
          const payload = await queryClient.fetchQuery({
            queryKey: keys.transactions.search({ q: queryString, csv: true }),
            queryFn: async () => {
              const res = await authFetch(
                apiUrl(`v1/search/agreements?${queryString}`),
              );
              if (!res.ok) throw new Error(`HTTP ${res.status}`);
              return (await res.json()) as TransactionSearchResponse;
            },
            staleTime: 60 * 1000,
          });
          rows = payload.results;
        } catch (error) {
          logger.error("Failed to fetch all transactions for CSV:", error);
          rows = results;
        }
      } else {
        rows = results;
      }

      if (rows.length === 0) return;

      const headers = [
        "Year",
        "Target",
        "Acquirer",
        "Deal Type",
        "Deal Status",
        "Attitude",
        "Purpose",
        "Consideration",
        "Deal Value",
        "Filed",
        "Announced",
        "Closed",
        "Target Industry",
        "Acquirer Industry",
        "SEC Filing URL",
        "Agreement UUID",
      ];

      const csv = [
        headers.join(","),
        ...rows.map((r) =>
          [
            r.year ?? "",
            csvEscape(r.target),
            csvEscape(r.acquirer),
            csvEscape(formatEnumValue(r.deal_type)),
            csvEscape(formatEnumValue(r.deal_status)),
            csvEscape(formatEnumValue(r.attitude)),
            csvEscape(formatEnumValue(r.purpose)),
            csvEscape(formatEnumValue(r.transaction_consideration)),
            csvEscape(formatCompactCurrencyValue(r.transaction_price_total)),
            csvEscape(formatDateValue(r.filing_date)),
            csvEscape(formatDateValue(r.announce_date)),
            csvEscape(formatDateValue(r.close_date)),
            csvEscape(r.target_industry),
            csvEscape(r.acquirer_industry),
            csvEscape(r.url),
            r.agreement_uuid,
          ].join(","),
        ),
      ].join("\n");

      const blob = new Blob([csv], { type: "text/csv;charset=utf-8;" });
      const link = document.createElement("a");
      const objectUrl = URL.createObjectURL(blob);
      link.href = objectUrl;
      const selectedText = selectedResults.size > 0 ? "_selected" : "";
      link.download = `ma_deals${selectedText}_${new Date().toISOString().split("T")[0]}.csv`;
      document.body.appendChild(link);
      link.click();
      link.remove();
      window.setTimeout(() => URL.revokeObjectURL(objectUrl), 0);
    },
    [results, selectedResults, queryClient],
  );

  const closeErrorModal = useCallback(() => {
    setShowErrorModal(false);
    setErrorMessage("");
  }, []);

  return {
    isSearching,
    results,
    hasSearched,
    totalCount,
    totalCountIsApproximate,
    totalPages,
    hasNext,
    hasPrev,
    access,
    showErrorModal,
    errorMessage,
    selectedResults,
    performSearch,
    clear,
    closeErrorModal,
    toggleResultSelection,
    toggleSelectAll,
    clearSelection,
    downloadCSV,
  };
}
