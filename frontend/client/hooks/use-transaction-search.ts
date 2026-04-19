import { useCallback, useState } from "react";
import { apiUrl } from "@/lib/api-config";
import { authFetch } from "@/lib/auth-fetch";
import { buildSearchParams } from "@/lib/url-params";
import { logger } from "@/lib/logger";
import type { ClauseTypeTree } from "@/lib/clause-types";
import type { SearchFilters } from "@shared/sections";
import type {
  TransactionSearchResponse,
  TransactionSearchResult,
} from "@shared/transactions";

export function useTransactionSearch() {
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
      if (markAsSearched) {
        setHasSearched(true);
      }

      try {
        const params = buildSearchParams(filters, clauseTypesNested);
        if (sortBy) {
          params.set("sort_by", sortBy);
          params.set("sort_direction", sortDirection);
        }
        const response = await authFetch(
          apiUrl(`v1/search/agreements?${params.toString()}`),
        );
        if (!response.ok) {
          throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }

        const payload = (await response.json()) as TransactionSearchResponse;
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
    [],
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
  }, []);

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
    performSearch,
    clear,
    closeErrorModal,
  };
}
