import { useCallback, useState } from "react";
import { useQueryClient } from "@tanstack/react-query";
import { apiUrl } from "@/lib/api-config";
import { authFetch } from "@/lib/auth-fetch";
import { buildSearchParams } from "@/lib/url-params";
import { logger } from "@/lib/logger";
import { LARGE_PAGE_SIZE_FOR_CSV, DEFAULT_PAGE } from "@/lib/constants";
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

export function useTransactionSearch() {
  const queryClient = useQueryClient();
  const [isSearching, setIsSearching] = useState(false);
  const [results, setResults] = useState<TransactionSearchResult[]>([]);
  const [hasSearched, setHasSearched] = useState(false);
  const [totalCount, setTotalCount] = useState(0);
  const [totalCountIsApproximate, setTotalCountIsApproximate] = useState(false);
  const [totalPages, setTotalPages] = useState(0);
  const [hasNext, setHasNext] = useState(false);
  const [hasPrev, setHasPrev] = useState(false);
  const [access, setAccess] = useState<TransactionSearchResponse["access"]>({
    tier: "anonymous",
  });
  const [errorMessage, setErrorMessage] = useState("");
  const [showErrorModal, setShowErrorModal] = useState(false);
  const [selectedResults, setSelectedResults] = useState<Set<string>>(
    new Set(),
  );

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
      setIsSearching(true);
      setShowErrorModal(false);
      setSelectedResults(new Set());
      if (markAsSearched) {
        setHasSearched(true);
      }

      try {
        const params = buildSearchParams(filters, clauseTypesNested);
        if (sortBy) {
          params.set("sort_by", sortBy);
          params.set("sort_direction", sortDirection);
        }
        const queryString = params.toString();
        const payload = await queryClient.fetchQuery({
          queryKey: keys.transactions.search({ q: queryString }),
          queryFn: async () => {
            const response = await authFetch(
              apiUrl(`v1/search/agreements?${queryString}`),
            );
            if (!response.ok) {
              throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }
            return (await response.json()) as TransactionSearchResponse;
          },
          staleTime: 60 * 1000,
        });
        setResults(payload.results);
        setAccess(payload.access);
        setTotalCount(payload.total_count);
        setTotalCountIsApproximate(payload.total_count_is_approximate);
        setTotalPages(payload.total_pages);
        setHasNext(payload.has_next);
        setHasPrev(payload.has_prev);
      } catch (error) {
        logger.error("Agreement search failed:", error);
        setResults([]);
        setAccess({ tier: "anonymous" });
        setTotalCount(0);
        setTotalCountIsApproximate(false);
        setTotalPages(0);
        setHasNext(false);
        setHasPrev(false);
        setErrorMessage(
          "Network error: unable to reach the back end database. Check your connection and try again.",
        );
        setShowErrorModal(true);
      } finally {
        setIsSearching(false);
      }
    },
    [queryClient],
  );

  const clear = useCallback(() => {
    setResults([]);
    setHasSearched(false);
    setTotalCount(0);
    setTotalCountIsApproximate(false);
    setTotalPages(0);
    setHasNext(false);
    setHasPrev(false);
    setAccess({ tier: "anonymous" });
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
          const params = buildSearchParams(
            searchFilters,
            clauseTypesNested,
            false,
          );
          params.append("page_size", LARGE_PAGE_SIZE_FOR_CSV.toString());
          params.append("page", DEFAULT_PAGE.toString());
          const res = await authFetch(
            apiUrl(`v1/search/agreements?${params.toString()}`),
          );
          if (!res.ok) throw new Error(`HTTP ${res.status}`);
          const payload = (await res.json()) as TransactionSearchResponse;
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
    [results, selectedResults],
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
